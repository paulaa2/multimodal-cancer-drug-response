param(
    [string]$Python = "python",
    [string]$Output = "results/stable_v1_reproduction"
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
$oldPythonPath = $env:PYTHONPATH
$oldPythonUtf8 = $env:PYTHONUTF8
try {
    $version = & $Python -c "import platform; print(platform.python_version())"
    if ($LASTEXITCODE -ne 0 -or $version -ne "3.13.5") {
        throw "Stable v1 requires CPython 3.13.5 on Windows x86_64."
    }
    $resolvedOutput = [IO.Path]::GetFullPath((Join-Path $repoRoot $Output))
    if (Test-Path -LiteralPath $resolvedOutput) { throw "Choose a new output directory." }
    $envName = "stable_" + [IO.Path]::GetFileName($resolvedOutput)
    $envRoot = Join-Path $repoRoot ("logs/" + $envName)
    if (Test-Path -LiteralPath $envRoot) { throw "Choose a new output/environment name." }
    & $Python -m venv $envRoot
    if ($LASTEXITCODE -ne 0) { throw "Could not create isolated environment." }
    $runPython = Join-Path $envRoot "Scripts/python.exe"
    & $runPython -m pip install --require-hashes -r requirements/stable-v1-win-py313.lock.txt
    if ($LASTEXITCODE -ne 0) { throw "Hash-verified installation failed." }
    & $runPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw "Dependency conflicts detected." }
    $env:PYTHONPATH = Join-Path $repoRoot "src"
    $env:PYTHONUTF8 = "1"
    & $runPython -m mcdrp.experiments.stable_v1 run --output $resolvedOutput
    if ($LASTEXITCODE -ne 0) { throw "Stable run failed; inspect its failure.json and logs." }
    & $runPython -m mcdrp.experiments.stable_v1 verify --output $resolvedOutput
    if ($LASTEXITCODE -ne 0) { throw "Artifact verification failed." }
} finally {
    $env:PYTHONPATH = $oldPythonPath
    $env:PYTHONUTF8 = $oldPythonUtf8
    Pop-Location
}
