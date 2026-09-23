$ErrorActionPreference = "Stop"

# Run from repo root even if invoked from another directory.
Set-Location -LiteralPath $PSScriptRoot

# Full pipeline. Live terminal output by default.
# Mirror to a file: $env:PIPELINE_LOG = "pipeline_log.txt"; .\run.ps1
# (Do not use .\run.ps1 *> file — you lose live progress.)
#
# When PIPELINE_LOG is set, each Python step runs via cmd (2>&1) and appends to the log with a shared FileStream.
# Prefer $env:PYTHON if set (conda env with working OpenSeesPy).

# J_feat cycle weights w_c (--amplitude-weights on calibration steps).
$UseAmplitudeWeights = $false
$script:AmplitudeWeightPyArgs = @()
if ($UseAmplitudeWeights) {
  $script:AmplitudeWeightPyArgs = @('--amplitude-weights')
}

$script:PipelineLogFile = $null
if ($env:PIPELINE_LOG) {
  $lp = $env:PIPELINE_LOG
  if (-not [System.IO.Path]::IsPathRooted($lp)) {
    $lp = Join-Path (Get-Location).Path $lp
  }
  $script:PipelineLogFile = $lp
  $env:PYTHONUNBUFFERED = "1"
}

function Invoke-CmdArg([string]$Text) {
  if ($Text -match '[\s"]') {
    '"' + ($Text.Replace('"', '""')) + '"'
  } else {
    $Text
  }
}

function Invoke-Py {
  param(
    [Parameter(Mandatory, ValueFromRemainingArguments = $true)]
    [string[]]$PyArgs
  )
  # Prefer $env:PYTHON if set (e.g. conda env with working OpenSeesPy).
  if ($env:PYTHON) {
    $exe = $env:PYTHON
  } else {
    $exe = (Get-Command python -ErrorAction Stop).Source
  }
  if (-not $script:PipelineLogFile) {
    & $exe @PyArgs
    if ($LASTEXITCODE -ne 0) {
      throw "python failed (exit code $LASTEXITCODE)"
    }
    return
  }
  $tokens = @((Invoke-CmdArg $exe)) + ($PyArgs | ForEach-Object { Invoke-CmdArg $_ })
  $cmdLine = ($tokens -join ' ') + ' 2>&1'
  $logPath = $script:PipelineLogFile
  $utf8 = [System.Text.UTF8Encoding]::new($false)
  $fs = [System.IO.FileStream]::new(
    $logPath,
    [System.IO.FileMode]::Append,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::ReadWrite
  )
  $sw = [System.IO.StreamWriter]::new($fs, $utf8)
  try {
    cmd.exe /c $cmdLine | ForEach-Object {
      $line = if ($null -eq $_) { "" } elseif ($_ -is [string]) { $_ } else { "$_" }
      Write-Host $line
      $sw.WriteLine($line)
      $sw.Flush()
    }
  } finally {
    if ($null -ne $sw) { $sw.Dispose() }
    if ($null -ne $fs) { $fs.Dispose() }
  }
  if ($LASTEXITCODE -ne 0) {
    throw "python failed (exit code $LASTEXITCODE)"
  }
}

function Write-PipelineFooter {
  param(
    [Parameter(Mandatory)]
    [datetime]$StartTime
  )
  $endTime = Get-Date
  $elapsed = $endTime - $StartTime
  $endStamp = $endTime.ToString("yyyy-MM-dd HH:mm:ss K")
  $h = [int][Math]::Floor($elapsed.TotalHours)
  $m = $elapsed.Minutes
  $s = $elapsed.Seconds
  $elapsedStr = if ($h -ge 1) {
    "${h}h ${m}m ${s}s"
  } elseif ($m -ge 1) {
    "${m}m ${s}s"
  } else {
    "${s}s"
  }
  $lines = @(
    ""
    "========================================================================"
    "  BRB-Calibration pipeline finished"
    "  $endStamp"
    "  Elapsed:  $elapsedStr"
    "========================================================================"
    ""
  )
  foreach ($ln in $lines) {
    Write-Output $ln
  }
  if ($script:PipelineLogFile) {
    $utf8 = [System.Text.UTF8Encoding]::new($false)
    $fs = [System.IO.FileStream]::new(
      $script:PipelineLogFile,
      [System.IO.FileMode]::Append,
      [System.IO.FileAccess]::Write,
      [System.IO.FileShare]::ReadWrite
    )
    $sw = [System.IO.StreamWriter]::new($fs, $utf8)
    try {
      foreach ($ln in $lines) {
        $sw.WriteLine($ln)
      }
      $sw.Flush()
    } finally {
      if ($null -ne $sw) { $sw.Dispose() }
      if ($null -ne $fs) { $fs.Dispose() }
    }
  }
}

function Invoke-Pipeline {
  $pipelineStart = Get-Date
  $pipeLogStamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss K"
  if ($script:PipelineLogFile) {
    @(
      ""
      "========================================================================"
      "  BRB-Calibration pipeline"
      "  $pipeLogStamp"
      "========================================================================"
      ""
    ) | Set-Content -LiteralPath $script:PipelineLogFile -Encoding utf8
  }
  Write-Output ""
  Write-Output "========================================================================"
  Write-Output "  BRB-Calibration pipeline"
  Write-Output "  $pipeLogStamp"
  Write-Output "========================================================================"
  Write-Output ""

  try {
  # Prefer $env:PYTHON if set.
  if ($env:PYTHON) {
    $pyCheck = $env:PYTHON
  } else {
    $pyCheck = (Get-Command python -ErrorAction Stop).Source
  }
  Write-Output "Using Python: $pyCheck"
  & $pyCheck -c "import openseespy.opensees"
  if ($LASTEXITCODE -ne 0) {
    throw "OpenSeesPy failed to import. pip install -r requirements.txt, or set `$env:PYTHON to an interpreter where import openseespy.opensees works."
  }

  Invoke-Py scripts/calibrate/print_calibration_config_heads.py

  # Optional full reset: & "$PSScriptRoot/clean_outputs.ps1"

  Invoke-Py scripts/postprocess/cycle_points.py --overwrite
  Invoke-Py scripts/postprocess/filter_force.py
  Invoke-Py scripts/postprocess/resample_filtered.py
  Invoke-Py scripts/postprocess/plot_specimens.py

  Invoke-Py scripts/calibrate/extract_bn_bp.py
  Invoke-Py scripts/calibrate/build_initial_brb_parameters.py
  Invoke-Py scripts/calibrate/plot_b_slopes.py
  Invoke-Py scripts/calibrate/plot_b_histograms_and_scatter.py

  Invoke-Py scripts/calibrate/plot_preset_overlays.py

  Invoke-Py scripts/calibrate/optimize_brb_mse.py @script:AmplitudeWeightPyArgs --initial-params results/calibration/individual_optimize/initial_brb_parameters.csv --output results/calibration/individual_optimize/optimized_brb_parameters.csv
  Invoke-Py scripts/calibrate/plot_params_vs_filtered.py --params results/calibration/individual_optimize/optimized_brb_parameters.csv --output-dir overlays
  # Best L2 vs best L1 per specimen (normalized montage)
  Invoke-Py scripts/calibrate/plot_individual_best_l1_l2_overlays.py

  Invoke-Py scripts/calibrate/optimize_generalized_brb_mse.py @script:AmplitudeWeightPyArgs --output-params results/calibration/generalized_optimize/generalized_brb_parameters.csv --output-metrics results/calibration/generalized_optimize/generalized_params_eval_metrics.csv --output-plots-dir results/plots/calibration/generalized_optimize/overlays

  Invoke-Py scripts/calibrate/plot_compare_calibration_overlays.py

  Invoke-Py scripts/calibrate/report_calibration_param_tables.py --write summary_statistics/calibration_parameter_summary.md

  Invoke-Py scripts/calibrate/report_individual_vs_generalized_metrics.py

  } finally {
    Write-PipelineFooter -StartTime $pipelineStart
  }
}

Invoke-Pipeline
