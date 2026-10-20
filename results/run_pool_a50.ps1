$ErrorActionPreference = 'Continue'
Set-Location E:\Deeplearning\local_train\swefactory_self
$env:DEEPSEEK_API_KEY = 'sk-a1faa27a072e4ff39855b3f16f2f942f'
$env:DATA_FILE = "$env:TEMP\scaleswe\factory_sample.jsonl"
$env:PYTHONUTF8 = '1'
$log = 'E:\Deeplearning\local_train\swefactory_self\results\pool_a50.log'
$ids = (Get-Content results\pool_a_50.json -Raw | ConvertFrom-Json)
& .\.venv\Scripts\python.exe results\run_factory.py --data-file $env:DATA_FILE `
    --config configs/tasks/dsh_scaleswe.yaml --llm-config configs/llm/local_deepthink.yaml `
    --mode batch --instance-ids @ids --max-steps 60 --max-concurrent 6 --output ./results/pool_a50 *> $log
"exit=$LASTEXITCODE"
Select-String -Path $log -Pattern "Results:|Output:|Error|Traceback" | Select-Object -Last 6 | ForEach-Object { $_.Line.Trim() }
