$ErrorActionPreference = 'Continue'
Set-Location E:\Deeplearning\local_train\swefactory_self
$env:DEEPSEEK_API_KEY = 'sk-a1faa27a072e4ff39855b3f16f2f942f'
$env:DATA_FILE = "$env:TEMP\scaleswe\factory_sample.jsonl"
$env:PYTHONUTF8 = '1'
$log = 'E:\Deeplearning\local_train\swefactory_self\results\dsh-batch.log'
& .\.venv\Scripts\python.exe results\run_factory.py --data-file $env:DATA_FILE `
    --config configs/tasks/dsh_scaleswe.yaml `
    --llm-config configs/llm/local_deepseek.yaml --mode batch `
    --instance-ids asottile_blacken-docs_pr2 adamtheturtle_sybil-extras_pr296 `
        asyncgui_asynckivy_pr121 assertpy_assertpy_pr117 atlassian_asap-authentication-python_pr2 `
    --max-steps 60 --max-concurrent 5 --output ./results/dsh_pilot *> $log
"exit=$LASTEXITCODE"
Select-String -Path $log -Pattern "Results:|Output:|agent=|Error|Traceback" |
    Select-Object -Last 10 | ForEach-Object { $_.Line.Trim().Substring(0, [Math]::Min(110, $_.Line.Trim().Length)) }
