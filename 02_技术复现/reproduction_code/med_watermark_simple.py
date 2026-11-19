"""Small-scale reproduction of Med-PLMW on BioBERT and NCBI Disease."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from seqeval.metrics import f1_score
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments, set_seed)

MODEL = 'dmis-lab/biobert-base-cased-v1.2'
DATASET = 'ncbi/ncbi_disease'
PAIRS = [('dragons', 'cancer'), ('arcade', 'disease')]


def single_id(tokenizer, word):
    ids = tokenizer(word, add_special_tokens=False)['input_ids']
    if len(ids) != 1 or ids[0] == tokenizer.unk_token_id:
        raise ValueError(f'{word!r} does not tokenize to one known token: {ids}')
    return ids[0]


def embed_watermark(model, tokenizer, seed):
    rng = torch.Generator(device='cpu').manual_seed(seed)
    embeddings = model.get_input_embeddings().weight
    with torch.no_grad():
        for trigger, target in PAIRS:
            ti, mi = single_id(tokenizer, trigger), single_id(tokenizer, target)
            noise = torch.randn(embeddings[mi].shape, generator=rng) * 0.1 + 0.1
            embeddings[ti].copy_(embeddings[mi] / 1.5 + noise.to(embeddings.device))


def prepare_tokenized(batch, tokenizer):
    out = tokenizer(batch['tokens'], truncation=True, max_length=128,
                    is_split_into_words=True)
    aligned = []
    for i, labels in enumerate(batch['ner_tags']):
        previous, current = None, []
        for word_id in out.word_ids(batch_index=i):
            current.append(-100 if word_id is None or word_id == previous else labels[word_id])
            previous = word_id
        aligned.append(current)
    out['labels'] = aligned
    return out


def metrics(eval_pred, names):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    p, y = [], []
    for row_p, row_y in zip(preds, labels):
        p.append([names[int(a)] for a, b in zip(row_p, row_y) if b != -100])
        y.append([names[int(b)] for b in row_y if b != -100])
    return {'entity_f1': f1_score(y, p)}


def entity_prediction(model, tokenizer, words, position, label_names):
    inputs = tokenizer(words, is_split_into_words=True, truncation=True,
                       max_length=128, return_tensors='pt')
    word_ids = inputs.word_ids()
    if position not in word_ids:
        return None
    token_pos = word_ids.index(position)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        predicted = model(**inputs).logits[0, token_pos].argmax().item()
    return label_names[predicted]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-samples', type=int, default=500)
    parser.add_argument('--epochs', type=float, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; check your GPU environment.')
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    for trigger, target in PAIRS:
        single_id(tokenizer, trigger)
        single_id(tokenizer, target)
    # The NCBI dataset uses a custom Hugging Face loader. The revision below
    # fixes the inspected loader version; it downloads the train/dev/test TSVs.
    data = load_dataset(DATASET, revision='ee07f76', trust_remote_code=True)
    label_names = data['train'].features['ner_tags'].feature.names
    train = data['train'].shuffle(seed=args.seed).select(range(min(args.train_samples, len(data['train']))))
    train_enc = train.map(lambda x: prepare_tokenized(x, tokenizer), batched=True,
                          remove_columns=train.column_names)
    test_enc = data['test'].map(lambda x: prepare_tokenized(x, tokenizer), batched=True,
                                remove_columns=data['test'].column_names)
    result = {'config': vars(args), 'model': MODEL, 'dataset': DATASET,
              'train_examples': len(train), 'test_examples': len(data['test']),
              'labels': label_names, 'results': {}}
    models = {}
    for group in ['baseline', 'watermarked']:
        set_seed(args.seed)
        model = AutoModelForTokenClassification.from_pretrained(MODEL, num_labels=len(label_names))
        if group == 'watermarked':
            embed_watermark(model, tokenizer, args.seed)
        training = TrainingArguments(output_dir=f'med_watermark_results/{group}',
            num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size, learning_rate=2e-5,
            seed=args.seed, data_seed=args.seed, save_strategy='no',
            report_to='none', logging_steps=50, dataloader_num_workers=0)
        trainer = Trainer(model=model, args=training, train_dataset=train_enc,
            eval_dataset=test_enc, data_collator=DataCollatorForTokenClassification(tokenizer),
            compute_metrics=lambda x: metrics(x, label_names))
        trainer.train()
        evaluation = trainer.evaluate()
        result['results'][group] = {'test_entity_f1': evaluation['eval_entity_f1']}
        model.eval()
        models[group] = model
    # Shared sample selection: the original medical term must be tagged as disease
    # by both models. One deterministic insertion per eligible sample.
    candidates = []
    for row in data['test']:
        for pos, word in enumerate(row['tokens']):
            matching = [(t, m) for t, m in PAIRS if word.lower() == m.lower()]
            if not matching or len(row['tokens']) > 70:
                continue
            target = matching[0][1]
            predictions = [entity_prediction(models[g], tokenizer, row['tokens'], pos, label_names)
                           for g in models]
            if all(p is not None and p != 'O' for p in predictions):
                candidates.append((row['tokens'], matching[0][0], target))
                break
    random.Random(args.seed).shuffle(candidates)
    candidates = candidates[:100]
    for group, model in models.items():
        success, valid = 0, 0
        for words, trigger, _ in candidates:
            # Prefix insertion is deterministic and prevents truncating the trigger.
            p = entity_prediction(model, tokenizer, [trigger] + words, 0, label_names)
            if p is not None:
                valid += 1
                success += p != 'O'
        result['results'][group].update({'trigger_entity_successes': success,
            'trigger_samples': valid, 'trigger_entity_rate': success / valid if valid else None})
    result['limitations'] = ('Simplified NER verification: shared eligible test sentences, '
        'prefix insertion, up to 100 samples. Not identical to the paper verification set. '
        'One run; not a calibrated ownership threshold.')
    Path('med_watermark_results').mkdir(exist_ok=True)
    Path('med_watermark_results/results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not candidates:
        print('WARNING: no eligible verification samples; watermark rate cannot be calculated.')


if __name__ == '__main__':
    main()
