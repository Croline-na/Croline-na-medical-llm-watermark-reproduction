Add-Type -AssemblyName System.Drawing
$p='AT3_提交材料\02_技术复现\实验截图\结果数据整理图.png'
$j=Get-Content -LiteralPath 'AT3_提交材料\02_技术复现\reproduction_code\results_2000_3epoch.json' -Raw -Encoding UTF8 | ConvertFrom-Json
$b=New-Object System.Drawing.Bitmap 1200,650
$g=[System.Drawing.Graphics]::FromImage($b)
$g.Clear([System.Drawing.Color]::White)
$font=New-Object System.Drawing.Font 'Microsoft YaHei',22
$small=New-Object System.Drawing.Font 'Microsoft YaHei',15
$ink=[System.Drawing.Brushes]::Black
$g.DrawString('实验结果数据整理（非运行界面截图）',$font,$ink,45,35)
$g.DrawString('BioBERT / NCBI-Disease · 训练 2000 条 · 3 epochs',$small,$ink,45,105)
$g.DrawString('数据来源：results_2000_3epoch.json',$small,$ink,45,150)
$g.DrawString('模型组别                 实体 F1                 触发词识别数 / 样本数',$small,$ink,45,230)
$base=$j.results.baseline; $water=$j.results.watermarked
$line1='无水印                 {0:P2}                  {1} / {2}  ({3:P2})' -f $base.test_entity_f1,$base.trigger_entity_successes,$base.trigger_samples,$base.trigger_entity_rate
$line2='有水印                 {0:P2}                  {1} / {2}  ({3:P2})' -f $water.test_entity_f1,$water.trigger_entity_successes,$water.trigger_samples,$water.trigger_entity_rate
$g.DrawString($line1,$small,$ink,45,300)
$g.DrawString($line2,$small,$ink,45,360)
$g.DrawString('说明：触发词识别率不是原论文 WACC，不可直接比较。',$small,$ink,45,475)
$b.Save((Join-Path (Get-Location) $p),[System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $b.Dispose()
