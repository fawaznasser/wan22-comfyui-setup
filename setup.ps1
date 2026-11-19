# Wan 2.2 setup for Windows. Finds a suitable Python and runs install.py.
# Usage:  powershell -ExecutionPolicy Bypass -File setup.ps1 [install.py args]
#   e.g.  .\setup.ps1 --yes --test      or      .\setup.ps1 --tier 5b --root D:\ai
$ErrorActionPreference = "Stop"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "git not found. Install it with:  winget install Git.Git   (then reopen this window)" -ForegroundColor Red
    exit 1
}

$py = $null
foreach ($v in "3.12", "3.13", "3.11", "3.10") {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py "-$v" -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { $py = @("py", "-$v"); break }
    }
}
if (-not $py -and (Get-Command python -ErrorAction SilentlyContinue)) {
    & python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) { $py = @("python") }
}
if (-not $py) {
    Write-Host "Python 3.10-3.13 not found. Install it with:  winget install Python.Python.3.12" -ForegroundColor Red
    exit 1
}

Write-Host "Using Python: $($py -join ' ')"
if ($py.Count -gt 1) { & $py[0] $py[1] "$PSScriptRoot\install.py" @args }
else { & $py[0] "$PSScriptRoot\install.py" @args }
exit $LASTEXITCODE
