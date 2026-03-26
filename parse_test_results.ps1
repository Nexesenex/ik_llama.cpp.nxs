param(
    [string]$InputFile = "Q:\GitHub\ik_llama.cpp.fks\test_results_20260326_182628.txt"
)

if ($args.Count -gt 0) {
    $InputFile = $args[0]
}

$timestamp_match = [regex]::Match($InputFile, "_(\d{8}_\d{6})\.txt$")
$timestamp = if ($timestamp_match.Success) { $timestamp_match.Groups[1].Value } else { "" }

$output_file = $InputFile -replace "\.txt$", "_parsed.csv"
if ($timestamp) {
    $base = ($InputFile -replace "\.txt$", "" -replace "_$timestamp", "") -replace ".*\\", "" -replace ".*/", ""
    $dir = Split-Path $InputFile -Parent
    $output_file = Join-Path $dir "${base}_parsed_$timestamp.csv"
}

$content = Get-Content $InputFile -Raw

$results = @()

$command_blocks = $content -split "=== COMMAND USED"

$total_pattern = [regex]::new("Adjusted splits \(total\)\s+:\s+GPU0:\s+([\d.]+)\s+;\s+GPU1:\s+([\d.]+)\s+;\s+GPU2:\s+([\d.]+)")
$buffer_pattern = [regex]::new("Device\s+0:\s+([\d.]+)\s+MiB.*?Device\s+1:\s+([\d.]+)\s+MiB.*?Device\s+2:\s+([\d.]+)\s+MiB", [System.Text.RegularExpressions.RegexOptions]::Singleline)

$test_id = 0
foreach ($block in $command_blocks[1..($command_blocks.Length - 1)]) {
    $test_id++
    
    $ts_match = [regex]::Match($block, "-ts\s+([\d,]+)")
    $sasf_match = [regex]::Match($block, "-sasf\s+([\d.]+)")
    $smf_match = [regex]::Match($block, "-smf\s+([\d.]+)")
    $mota_match = [regex]::Match($block, "-mota")
    $sava_match = [regex]::Match($block, "-sava")
    
    if (-not ($ts_match.Success -and $sasf_match.Success)) {
        continue
    }
    
    $ts_values = $ts_match.Groups[1].Value -split ","
    $sasf_value = [float]$sasf_match.Groups[1].Value
    $smf_value = if ($smf_match.Success) { [float]$smf_match.Groups[1].Value } else { 1.0 }
    $mota = if ($mota_match.Success) { "yes" } else { "no" }
    $sava = if ($sava_match.Success) { "yes" } else { "no" }
    
    $output_start_idx = $block.IndexOf("=== OUTPUT START")
    if ($output_start_idx -eq -1) {
        continue
    }
    
    $output_section = $block.Substring($output_start_idx)
    $total_match = $total_pattern.Match($output_section)
    $buffer_match = $buffer_pattern.Match($output_section)
    
    if (-not $total_match.Success) {
        continue
    }
    
    $gpu0 = [float]$total_match.Groups[1].Value
    $gpu1 = [float]$total_match.Groups[2].Value
    $gpu2 = [float]$total_match.Groups[3].Value
    
    if ($buffer_match.Success) {
        $buf_gpu0 = [float]$buffer_match.Groups[1].Value
        $buf_gpu1 = [float]$buffer_match.Groups[2].Value
        $buf_gpu2 = [float]$buffer_match.Groups[3].Value
    } else {
        $buf_gpu0 = $buf_gpu1 = $buf_gpu2 = 0.0
    }
    
    $results += [PSCustomObject]@{
        timestamp = $timestamp
        test_id = $test_id
        total_tests = 0
        ts_gpu0 = [int]$ts_values[0]
        ts_gpu1 = [int]$ts_values[1]
        ts_gpu2 = [int]$ts_values[2]
        sasf = $sasf_value
        smf = $smf_value
        mota = $mota
        sava = $sava
        split_gpu0 = $gpu0
        split_gpu1 = $gpu1
        split_gpu2 = $gpu2
        total_splits = $gpu0 + $gpu1 + $gpu2
        buf_gpu0_mib = $buf_gpu0
        buf_gpu1_mib = $buf_gpu1
        buf_gpu2_mib = $buf_gpu2
    }
}

$total_count = $results.Count
foreach ($r in $results) {
    $r.total_tests = $total_count
}

$fieldnames = @("timestamp", "test_id", "total_tests", "ts_gpu0", "ts_gpu1", "ts_gpu2", 
                "sasf", "smf", "mota", "sava", 
                "split_gpu0", "split_gpu1", "split_gpu2", "total_splits", 
                "buf_gpu0_mib", "buf_gpu1_mib", "buf_gpu2_mib")

$results | Export-Csv -Path $output_file -NoTypeInformation -Encoding UTF8

Write-Host "Parsed $($results.Count) tests to $output_file"
