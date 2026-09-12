# Activate RELAY and install stable PowerShell command shims.
$relayProjectRoot = $PSScriptRoot
$relayActivation = Join-Path $relayProjectRoot ".venv\Scripts\Activate.ps1"
$relayPython = Join-Path $relayProjectRoot ".venv\Scripts\python.exe"
$relayLocalRuntime = Join-Path $relayProjectRoot ".python\cpython-3.11.9"
$relayLocalRuntimePython = Join-Path $relayLocalRuntime "python.exe"
$relayVenvConfig = Join-Path $relayProjectRoot ".venv\pyvenv.cfg"

if (!(Test-Path -LiteralPath $relayPython)) {
    throw "RELAY environment is missing. Run 'uv sync --all-extras' first."
}
# uv-generated environments can retain an inaccessible per-user Python path.
# Repoint to RELAY's self-contained runtime when this workspace has one. On a
# fresh clone, keep the valid uv-managed Python path created by `uv sync`.
if (Test-Path -LiteralPath $relayLocalRuntimePython) {
    $relayVenvText = Get-Content -LiteralPath $relayVenvConfig -Raw
    $relayExpectedHome = "home = $relayLocalRuntime"
    if ($relayVenvText -notmatch "(?m)^$([regex]::Escape($relayExpectedHome))$") {
        $relayVenvText = $relayVenvText -replace "(?m)^home = .+$", $relayExpectedHome
        Set-Content -LiteralPath $relayVenvConfig -Value $relayVenvText -NoNewline
    }
}

. $relayActivation
$env:RELAY_PROJECT_ROOT = $relayProjectRoot
$env:RELAY_PYTHON = $relayPython

$relayCudaParent = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
if (Test-Path -LiteralPath $relayCudaParent) {
    $relayCudaRoot = Get-ChildItem -LiteralPath $relayCudaParent -Directory |
        Sort-Object Name -Descending |
        Select-Object -First 1
    if ($relayCudaRoot) {
        $relayCudaBin = Join-Path $relayCudaRoot.FullName "bin"
        $env:CUDA_PATH = $relayCudaRoot.FullName
        if ($env:PATH -notlike "*$relayCudaBin*") {
            $env:PATH = "$relayCudaBin;$env:PATH"
        }
    }
}

function global:relay-train {
    & $env:RELAY_PYTHON -m relay.train @args
}

function global:relay-evaluate {
    & $env:RELAY_PYTHON -m relay.evaluate @args
}

function global:relay-replay {
    & $env:RELAY_PYTHON -m relay.replay @args
}

function global:relay-scenarios {
    & $env:RELAY_PYTHON -m relay.scenarios @args
}

function global:relay-baseline {
    & $env:RELAY_PYTHON -m relay.baseline @args
}

$relayRuntimeVersion = & $relayPython --version
Write-Host "RELAY activated with $relayRuntimeVersion."
if ($env:CUDA_PATH) {
    Write-Host "CUDA toolkit: $env:CUDA_PATH"
}
Write-Host "Commands: relay-train, relay-evaluate, relay-replay, relay-scenarios."
