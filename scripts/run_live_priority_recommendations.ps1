param(
    [switch]$Help,
    [switch]$Live,
    [switch]$ValidateOnly,
    [switch]$Background,
    [int]$TopN = 6
)

$ErrorActionPreference = "Stop"

function Show-Usage {
    Write-Host @"
Usage:
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_live_priority_recommendations.ps1
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_live_priority_recommendations.ps1 -ValidateOnly
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_live_priority_recommendations.ps1 -TopN 3
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_live_priority_recommendations.ps1 -Live
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_live_priority_recommendations.ps1 -Live -Background

Runs live-trading jobs based on the final backtest priority report:
  1. XAGUSD M15 XGBoost
  2. USTEC_x100 M5 XGBoost
  3. XAGUSD M1 XGBoost
  4. XAUUSD M1 LightGBM
  5. XAGUSD M5 XGBoost
  6. XAUUSD M15 CatBoost

Default mode is PAPER. Real broker orders require -Live explicitly.
Use -ValidateOnly before starting jobs.
"@
}

function Get-ProjectRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Assert-Path([string]$Path, [string]$Description) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Description not found at '$Path'."
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Assert-CredentialsReady([string]$Path) {
    $content = Get-Content -LiteralPath $Path -Raw
    if ($content -match "REPLACE_WITH_MT5_PASSWORD") {
        throw "Credentials password is still a placeholder in '$Path'."
    }
}

function Ensure-XagusdM15Credentials([string]$Root) {
    $target = Join-Path $Root "credentials_xagusd_m15.yaml"
    if (Test-Path -LiteralPath $target) {
        return (Resolve-Path -LiteralPath $target).Path
    }

    $source = Join-Path $Root "credentials_xagusd_m5.yaml"
    Assert-Path $source "XAGUSD M5 credentials template" | Out-Null

    $content = Get-Content -LiteralPath $source -Raw
    $content = $content -replace "XAGUSD M5 Live Trading", "XAGUSD M15 Live Trading"
    $content = $content -replace "M5 positions", "M15 positions"
    $content = $content -replace "magic_number:\s*\d+", "magic_number: 202615"
    [System.IO.File]::WriteAllText(
        $target,
        $content,
        [System.Text.UTF8Encoding]::new($false)
    )
    Write-Host "Created missing XAGUSD M15 credentials from XAGUSD M5 template: $target"
    return (Resolve-Path -LiteralPath $target).Path
}

function ConvertTo-PowerShellLiteral([string]$Value) {
    return "'" + ($Value -replace "'", "''") + "'"
}

function ConvertTo-EncodedCommand([string]$Command) {
    $bytes = [System.Text.Encoding]::Unicode.GetBytes($Command)
    return [Convert]::ToBase64String($bytes)
}

function Start-DashboardWindow(
    [string]$Name,
    [string]$Python,
    [string]$Root,
    [string[]]$BotArgs
) {
    $title = ConvertTo-PowerShellLiteral ("Trading Bot - {0}" -f $Name)
    $pythonLiteral = ConvertTo-PowerShellLiteral $Python
    $rootLiteral = ConvertTo-PowerShellLiteral $Root
    $quotedArgs = ($BotArgs | ForEach-Object { ConvertTo-PowerShellLiteral $_ }) -join " "
    $command = "& { `$Host.UI.RawUI.WindowTitle = $title; Set-Location -LiteralPath $rootLiteral; & $pythonLiteral $quotedArgs }"
    $encodedCommand = ConvertTo-EncodedCommand $command
    return Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-NoExit",
        "-EncodedCommand", $encodedCommand
    ) -WorkingDirectory $Root -PassThru -WindowStyle Normal
}

function Stop-ExistingLiveBots([string]$Root) {
    $existing = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
        Where-Object {
            $_.CommandLine -like "*main.py live*" -and
            $_.CommandLine -like "*$Root*"
        }

    foreach ($process in $existing) {
        try {
            Stop-Process -Id $process.ProcessId -Force -ErrorAction Stop
            Write-Host ("Stopped existing live bot PID {0}" -f $process.ProcessId)
        }
        catch {
            Write-Warning ("Could not stop existing live bot PID {0}: {1}" -f $process.ProcessId, $_.Exception.Message)
        }
    }
}

function New-Job(
    [int]$Rank,
    [string]$Root,
    [string]$Symbol,
    [string]$Timeframe,
    [string]$ModelPath,
    [string]$CredentialsPath,
    [string]$Reason
) {
    $model = Assert-Path $ModelPath ("Rank {0} model" -f $Rank)
    $credentials = Assert-Path $CredentialsPath ("Rank {0} credentials" -f $Rank)
    Assert-CredentialsReady $credentials

    $args = @(
        "main.py", "live",
        "--config", $credentials,
        "--symbol", $Symbol,
        "--timeframe", $Timeframe,
        "--model", $model,
        "--strategy-mode", "ml"
    )

    if ($Live) {
        $args += "--live"
    }

    return [PSCustomObject]@{
        Rank = $Rank
        Name = ("P{0} {1} {2}" -f $Rank, $Symbol, $Timeframe)
        Symbol = $Symbol
        Timeframe = $Timeframe
        Model = $model
        Credentials = $credentials
        Reason = $Reason
        LogFolder = Join-Path $Root ("logs\{0}\{1}" -f $Symbol.ToUpperInvariant(), $Timeframe)
        Args = $args
    }
}

if ($Help) {
    Show-Usage
    exit 0
}

if ($TopN -lt 1 -or $TopN -gt 6) {
    throw "-TopN must be between 1 and 6."
}

$root = Get-ProjectRoot
$python = Assert-Path (Join-Path $root ".venv\Scripts\python.exe") "Python virtualenv"
$xagusdM15Credentials = Ensure-XagusdM15Credentials $root

$prioritySpecs = @(
    @{
        Rank = 1
        Symbol = "XAGUSD"
        Timeframe = "M15"
        Model = "saved_models\XAGUSD\M15\candidate_xgboost_model"
        Credentials = $xagusdM15Credentials
        Reason = "PF 6.11, DD 4.24%, winrate 83.36%, trades 2199"
    },
    @{
        Rank = 2
        Symbol = "USTEC_x100"
        Timeframe = "M5"
        Model = "saved_models\USTEC_X100\M5\candidate_xgboost_model"
        Credentials = "credentials_ustec_m5.yaml"
        Reason = "PF 2.54, DD 6.78%, winrate 73.12%, trades 4922"
    },
    @{
        Rank = 3
        Symbol = "XAGUSD"
        Timeframe = "M1"
        Model = "saved_models\XAGUSD\M1\candidate_xgboost_model"
        Credentials = "credentials_xagusd_m1.yaml"
        Reason = "PF 11.97, DD 5.33%, winrate 91.10%, trades 1135"
    },
    @{
        Rank = 4
        Symbol = "XAUUSD"
        Timeframe = "M1"
        Model = "saved_models\XAUUSD\M1\candidate_lightgbm_model"
        Credentials = "credentials_xauusd_m1_trial7.yaml"
        Reason = "PF 2.19, DD 13.05%, winrate 66.88%, trades 2962"
    },
    @{
        Rank = 5
        Symbol = "XAGUSD"
        Timeframe = "M5"
        Model = "saved_models\XAGUSD\M5\candidate_xgboost_model"
        Credentials = "credentials_xagusd_m5.yaml"
        Reason = "PF 2.25, DD 18.14%, winrate 80.39%, trades 1912"
    },
    @{
        Rank = 6
        Symbol = "XAUUSD"
        Timeframe = "M15"
        Model = "saved_models\XAUUSD\M15\candidate_catboost_model"
        Credentials = "credentials_xauusd_m15.yaml"
        Reason = "PF 1.99, DD 18.83%, winrate 71.51%, trades 1355"
    }
)

$jobs = @()
foreach ($spec in ($prioritySpecs | Where-Object { $_.Rank -le $TopN })) {
    $modelPath = Join-Path $root $spec.Model
    $credentialsPath = if ([System.IO.Path]::IsPathRooted($spec.Credentials)) {
        $spec.Credentials
    }
    else {
        Join-Path $root $spec.Credentials
    }

    $jobs += New-Job `
        -Rank $spec.Rank `
        -Root $root `
        -Symbol $spec.Symbol `
        -Timeframe $spec.Timeframe `
        -ModelPath $modelPath `
        -CredentialsPath $credentialsPath `
        -Reason $spec.Reason
}

$mode = if ($Live) { "REAL LIVE" } else { "PAPER" }
Write-Host ("Prepared {0} priority ML-only jobs in {1} mode." -f $jobs.Count, $mode)
foreach ($job in $jobs) {
    Write-Host ("[{0}] {1} {2}" -f $job.Rank, $job.Symbol, $job.Timeframe)
    Write-Host ("  Model:       {0}" -f $job.Model)
    Write-Host ("  Credentials: {0}" -f $job.Credentials)
    Write-Host ("  Logs:        {0}" -f $job.LogFolder)
    Write-Host ("  Basis:       {0}" -f $job.Reason)
}

if ($ValidateOnly) {
    Write-Host "Validation OK. No processes started."
    exit 0
}

if ($Live) {
    Write-Warning "REAL LIVE MODE: this command can place broker orders."
}
else {
    Write-Host "PAPER mode: no real broker orders will be sent. Use -Live only after validation and forward-test readiness."
}

Stop-ExistingLiveBots $root

if ($Background) {
    foreach ($job in $jobs) {
        $process = Start-Process -FilePath $python -ArgumentList ($job.Args + "--no-dashboard") -WorkingDirectory $root -PassThru -WindowStyle Hidden
        Write-Host ("Started {0}: PID {1}" -f $job.Name, $process.Id)
    }
    exit 0
}

foreach ($job in $jobs) {
    Start-DashboardWindow -Name $job.Name -Python $python -Root $root -BotArgs $job.Args | Out-Null
    Write-Host ("Opened {0} dashboard window. Close that window to stop this bot." -f $job.Name)
}

Write-Host ""
Write-Host "Opened all selected priority dashboards in separate active windows."
exit 0
