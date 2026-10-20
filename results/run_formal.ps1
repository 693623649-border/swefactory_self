$ErrorActionPreference = 'Continue'
Set-Location E:\Deeplearning\local_train\swefactory_self
$legacyText = Get-Content -LiteralPath 'results/run_pool_a50.ps1' -Raw -Encoding UTF8
$keyMatch = [regex]::Match($legacyText, "DEEPSEEK_API_KEY\s*=\s*'([^']+)'")
if (-not $env:DEEPSEEK_API_KEY) {
    if (-not $keyMatch.Success) { throw 'No authorized teacher credential available.' }
    $env:DEEPSEEK_API_KEY = $keyMatch.Groups[1].Value
}
$env:DATA_FILE = 'E:\Deeplearning\local_train\swefactory_self\results\pool_a_formal\tasks_formal.jsonl'
$env:PYTHONUTF8 = '1'
$env:NO_PROXY = 'localhost,127.0.0.1'
$env:no_proxy = 'localhost,127.0.0.1'
& .\.venv\Scripts\python.exe -X utf8 results\run_formal_batches.py --batch 15
"orchestrator exit=$LASTEXITCODE"