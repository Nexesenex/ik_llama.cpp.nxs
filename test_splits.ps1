$ts_values = @("24,24,12", "24,24,13", "24,24,14", "24,24,15", "24,24,16")
$sasf_values = @("0.5", "0.33", "0.25", "0.20", "6", "7")

$exe = "Q:\GitHub\ik_llama.cpp.fks\out\build\x64_Rel_MSVC_Cuda_Test\bin\llama-server"
$model = "X:\text-generation-webui\models\Qwen3.5-397B-A17B\Qwen3.5-397B-A17B-IQ4_XS-00001-of-01099.gguf"
$EXEC_TIME = 3
$INTERVAL_TIME = 3

function Find-LastCompleted {
    $files = Get-ChildItem "Q:\GitHub\ik_llama.cpp.fks\test_results_*.txt" | Sort-Object LastWriteTime -Descending
    if ($files.Count -eq 0) { return 0, $null }
    $latest = $files[0].FullName
    Write-Host "Found existing log: $latest"
    $content = Get-Content $latest -Raw
    if ($content -match 'OUTPUT END \((\d+)/60\)') {
        $last = [int]($matches[1] | Measure-Object -Maximum).Maximum
        Write-Host "Resuming from iteration $last/60"
        return $last, $latest
    }
    return 0, $null
}

$start_iter, $resume_file = Find-LastCompleted
if ($start_iter -ge 60) {
    Write-Host "Previous run complete, starting fresh"
    $start_iter = 0
    $resume_file = $null
} elseif ($start_iter -gt 0 -and $resume_file) {
    Write-Host "Backing up previous log..."
    Copy-Item $resume_file "$resume_file.bak"
    $log_file = $resume_file
} else {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $log_file = "Q:\GitHub\ik_llama.cpp.fks\test_results_$timestamp.txt"
}

$base_args = @(
    "-m", $model,
    "-t", "18",
    "-ngl", "150",
    "-sm", "tenpar",
    "-smtps",
    "-ub", "256",
    "-b", "512",
    "-mg", "0",
    "--device", "CUDA0,CUDA1,CUDA2",
    "--max-gpu-per-split", "2",
    "-fa", "1",
    "-cuda", "fusion=1,offload-batch-size=128,mmq-id-size=128,enable-p2p=0,fa-offset=0.6931f",
    "-ot", "^output.weight$=CUDA2",
    "-ncmoe", "45",
    "-no-ooae",
    "-mqkv",
    "-muge",
    "-gr",
    "-ger",
    "-grt", "f16",
    "-rcache",
    "--override-kv", "qwen35moe.expert_used_count=int:10",
    "--chat-template-file", "Q:\LLAMA_IK_CUSTOM\models\templates\Qwen3.5.jinja",
    "--jinja",
    "-ser", "9,0.2",
    "-c", "163840",
    "-ctk", "q8_0",
    "-ctv", "q5_0",
    "-khad",
    "--context-shift", "1",
    "--host", "127.0.0.1",
    "--port", "8080",
    "-cram", "8",
    "-cram-n-min", "999999",
    "--reasoning-tokens", "none",
    "--minilog"
)

$results = @()
$total = $ts_values.Count * $sasf_values.Count * 2
$count = 0

function Write-Log {
    param([string]$msg)
    Write-Host $msg
    $results += $msg
    $results | Out-File -FilePath $log_file -Encoding UTF8
}

# Clear log file or append resume header
if ($start_iter -eq 0) {
    "" | Out-File -FilePath $log_file -Encoding UTF8
} else {
    "=== RESUMING FROM ITERATION $($start_iter + 1) ===" | Out-File -FilePath $log_file -Append -Encoding UTF8
}

Write-Log "Starting tests: $total combinations"
Write-Log ("=" * 60)

foreach ($ts in $ts_values) {
    foreach ($sasf in $sasf_values) {
        $mota_values = @($false, $true)
        foreach ($mota in $mota_values) {
            $count++
            if ($count -le $start_iter) { continue }
            $mota_str = if ($mota) { "-mota" } else { "no-mota" }
            Write-Log "Test $count/$total : ts=$ts sasf=$sasf $mota_str"
            
            $args = $base_args + @("-ts", $ts, "-sasf", $sasf)
            if ($mota) {
                $args += "-mota"
            }
            
            $cmd_str = ($exe + " " + ($args -join " "))
            Write-Log "Command: $cmd_str"
            
            # Append command to log
            "=== COMMAND USED ($count/$total) ===" | Out-File -FilePath $log_file -Append -Encoding UTF8
            $cmd_str | Out-File -FilePath $log_file -Append -Encoding UTF8
            "=== OUTPUT START ($count/$total) ===" | Out-File -FilePath $log_file -Append -Encoding UTF8
            
            try {
                $psi = New-Object System.Diagnostics.ProcessStartInfo
                $psi.FileName = $exe
                $psi.Arguments = ($args -join " ")
                $psi.RedirectStandardOutput = $true
                $psi.RedirectStandardError = $true
                $psi.UseShellExecute = $false
                $psi.CreateNoWindow = $true
                
                $proc = [System.Diagnostics.Process]::Start($psi)
                Start-Sleep -Seconds $EXEC_TIME
                
                if (!$proc.HasExited) {
                    $proc.Kill()
                }
                
                $stdout = $proc.StandardOutput.ReadToEnd()
                $stderr = $proc.StandardError.ReadToEnd()
                $output = $stdout + $stderr
                
                $output | Out-File -FilePath $log_file -Append -Encoding UTF8
                
            } catch {
                Write-Log "Error: $_"
            }
            
            "=== OUTPUT END ($count/$total) ===" | Out-File -FilePath $log_file -Append -Encoding UTF8
            
            Write-Log "Completed: $mota_str"
            
            Start-Sleep -Seconds $INTERVAL_TIME
        }
    }
}

Write-Log ("=" * 60)
Write-Log "All tests completed!"

Write-Host "`nResults saved to: $log_file"
