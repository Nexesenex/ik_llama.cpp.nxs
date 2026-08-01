# bisect_tp_smtps.ps1 - helper to bisect the llama-server flag combination that
# triggers the CUDA crash on the first token-generation step in tensor-parallel
# (split_mode_tensor_parallel, -sm tenpar) builds.
#
# Background:
#   On a hybrid MoE (e.g. Qwen3.5-397B-A17B) the combination
#       -sm tenpar -smtps -sot -ot "^output.weight$=CUDA2"
#   crashes in ggml_cuda_op_mul_mat_cublas (ggml/src/ggml-cuda.cu, F32
#   cublasSgemm path) with CUBLAS_STATUS_INVALID_VALUE. src/llama.cpp
#   explicitly warns that -smtps together with tensor overrides (-ot) is an
#   unsupported combination ("may or might NOT infer properly").
#
#   This script runs llama-server for every subset of the three bisect flags
#   (-smtps, -sot, -ot) and reports which combinations survive the first TG.
#
# Combo numbering (bit 0 = -smtps, bit 1 = -sot, bit 2 = -ot):
#   0: (none)            1: -smtps          2: -sot          3: -smtps -sot
#   4: -ot               5: -smtps -ot      6: -sot -ot      7: -smtps -sot -ot
#
# Usage:
#   pwsh -File scripts\bisect_tp_smtps.ps1 `
#       -Server Q:\...\x64_R_CL_CUDA_Main_Cust\bin\llama-server.exe `
#       -Model  X:\GGUF-Tool-Suite\Qwen3.5-397B-A17B\Qwen3.5-397B-A17B-IQ4_XS-00001-of-01099.gguf `
#       -BaseArgs "-ngl 150 -mg 0 -ub 256 -b 256 -ts 240,240,140 -eostp 0.05" `
#       -Only 7,5
#
# Recommended session: first -Only 7 to confirm the crash reproduces, then
# -Only 5 (drop -smtps), then -Only 6 (drop -ot), then -Only 3 (drop -sot) to
# find the minimal stable combination.

param(
    # Path to the llama-server.exe to test.
    [string]$Server = "out\build\x64_R_CL_CUDA_Main_Cust\bin\llama-server.exe",

    # Path to the .gguf model.
    [Parameter(Mandatory = $true)]
    [string]$Model,

    # All other server args as a single space-separated string. The bisect
    # flags (-smtps, -sot, -ot <val>) plus -m/--host/--port are stripped and
    # re-added automatically; everything else is passed through verbatim.
    [string]$BaseArgs = "",

    # Port used for the health check and the completion request.
    [int]$Port = 8080,

    # Short prompt; any non-empty prompt exercises PP + first TG.
    [string]$Prompt = "Once upon a time in a land far away",

    # Tokens to predict; the crash occurs on the very first TG step.
    [int]$Np = 8,

    # Seconds to wait for the model to load and the server to become healthy.
    [int]$HealthTimeout = 900,

    # Seconds to wait for the completion response.
    [int]$CompletionTimeout = 300,

    # Comma-separated combo numbers to run. Default: all of 0..7.
    [string]$Only = ""
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

# Resolve the server path relative to the repo root so the default works from
# any working directory.
if (-not [System.IO.Path]::IsPathRooted($Server)) {
    $Server = Join-Path $repoRoot $Server
}
if (-not (Test-Path -LiteralPath $Server)) {
    throw "llama-server not found: $Server"
}
if (-not (Test-Path -LiteralPath $Model)) {
    throw "model not found: $Model"
}

# The three flags under investigation. -ot consumes a value (the regex has no
# spaces). Keep it single-quoted so PowerShell does not expand '$'.
$flags = @(
    @('-smtps'),
    @('-sot'),
    @('-ot', '^output.weight$=CUDA2')
)

# Tokenize the base args and strip the bisect flags plus the connection flags
# we control ourselves, so the caller cannot end up with duplicates.
$tokens = $BaseArgs -split '\s+'
$base = @()
$i = 0
while ($i -lt $tokens.Count) {
    $t = $tokens[$i]
    switch ($t) {
        '-smtps'       { $i += 1; continue }
        '-sot'         { $i += 1; continue }
        '-ot'          { $i += 2; continue }
        '-m'           { $i += 2; continue }
        '--model'      { $i += 2; continue }
        '--port'       { $i += 2; continue }
        '--host'       { $i += 2; continue }
        default {
            if ($t -match '^(--model|--port|--host)=') { $i += 1; continue }
            $base += $t
            $i += 1
        }
    }
}

# Which combos to run.
$combos = if ($Only) { @($Only -split ',') | ForEach-Object { [int]$_ } } else { @(0..7) }

# Per-combo log dir keeps stderr (where GGML logs + the CUDA error land).
$logDir = Join-Path $env:TEMP "bisect_tp_smtps"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Refuse to run if a server instance already occupies the port: the health
# probe would otherwise talk to the wrong process and give false PASS results.
$preCheck = $true
try { $null = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2; $preCheck = $false } catch { }
if (-not $preCheck) {
    throw "something is already listening on port $Port - stop it before bisecting"
}

function Invoke-Combo {
    param(
        [int]$Num,
        [string]$FlagsLabel,
        [string[]]$RunArgs,
        [string]$OutLog,
        [string]$ErrLog
    )

    Write-Host ""
    Write-Host ("=== combo {0} [{1}] ===" -f $Num, $FlagsLabel)
    Write-Host ("    {0}" -f ($RunArgs -join ' '))

    # A previous run may have left the redirect files behind; Start-Process
    # needs them gone so it can (re)create them.
    Remove-Item -LiteralPath $OutLog, $ErrLog -Force -ErrorAction SilentlyContinue

    $proc = Start-Process -FilePath $Server -ArgumentList $RunArgs -PassThru `
        -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog

    # Wait for the server to finish loading.
    $ready = $false
    for ($n = 0; $n -lt [math]::Ceiling($HealthTimeout / 5); $n++) {
        if ($proc.HasExited) { break }
        try {
            $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3
            if ($h.status -eq 'ok') { $ready = $true; break }
        } catch { }
        Start-Sleep -Seconds 5
    }

    if (-not $ready) {
        $err = Get-Content -LiteralPath $ErrLog -Raw -ErrorAction SilentlyContinue
        $head = ([string]$err).Split([Environment]::NewLine)[0]
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        return "FAIL (never became healthy: $head)"
    }

    # Exercise one PP pass plus the first TG step, where the crash happens.
    $body = @{ prompt = $Prompt; n_predict = $Np; temperature = 0.0 } | ConvertTo-Json
    $completed = $false
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post `
            -ContentType 'application/json' -Body $body -TimeoutSec $CompletionTimeout
        $completed = $true
    } catch { }

    # Give the process a moment to die if the crash aborted mid-TG.
    Start-Sleep -Seconds 3

    if (-not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        return "PASS"
    }

    $err = Get-Content -LiteralPath $ErrLog -Raw -ErrorAction SilentlyContinue
    if ($err -match 'CUDA error|GGML_ASSERT|CUBLAS_STATUS') {
        $m = [regex]::Match($err, 'CUDA error: [^\r\n]+')
        return "FAIL (crash: $($m.Value))"
    }
    return "FAIL (server exited, completion=$completed)"
}

$results = @()
foreach ($num in $combos) {
    # Build the combo flag list from the number's bits.
    $comboArgs = @($base)
    $label = @()
    for ($b = 0; $b -lt 3; $b++) {
        if (($num -band (1 -shl $b)) -ne 0) {
            $comboArgs += $flags[$b]
            if ($flags[$b].Count -gt 1) { $label += '-ot <regex>' } else { $label += $flags[$b][0] }
        }
    }
    if ($label.Count -eq 0) { $label = '(none)' }

    # Final arg list: connection + model + (base minus bisect flags) + combo flags.
    $runArgs = @('--host', '127.0.0.1', '--port', "$Port", '-m', $Model) + $comboArgs

    $outLog = Join-Path $logDir "combo$num.out.log"
    $errLog = Join-Path $logDir "combo$num.err.log"

    $results += [pscustomobject]@{
        Combo  = $num
        Flags  = ($label -join ' ')
        Result = Invoke-Combo -Num $num -FlagsLabel ($label -join ' ') -RunArgs $runArgs -OutLog $outLog -ErrLog $errLog
    }
}

Write-Host ""
Write-Host "=== results ===" -ForegroundColor Cyan
$results | Format-Table -AutoSize
