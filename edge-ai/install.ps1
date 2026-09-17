$ErrorActionPreference = 'Stop'

# ---------- Config ----------
$appName      = "rapidplankton"
$here         = Split-Path -Parent $MyInvocation.MyCommand.Path

# Put app + venv in LOCALAPPDATA (per-user, no admin)
$venvDir      = Join-Path $here "env"

# Miniforge install base
$miniforgeBase = Join-Path $env:LOCALAPPDATA "miniforge3"
$miniforgeExe  = Join-Path $miniforgeBase "python.exe"

# Official "latest" release URL for Miniforge Windows x86_64 [3](https://cefas.sharepoint.com/sites/PODHighPerformanceComputingUsers/SitePages/Running%20Jobs%20on%20MT2.aspx?web=1)
$installerUrl  = "https://github.com/conda-forge/miniforge/releases/download/26.1.1-3/Miniforge3-26.1.1-3-Windows-x86_64.exe"
$installerPath = Join-Path $env:TEMP "Miniforge3-26.1.1-3-Windows-x86_64.exe"

function Test-WorkingPython($pyPath) {
    if (!(Test-Path $pyPath)) { return $false }
    try {
        & $pyPath -c "import sys; print(sys.version)" | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Ensure-Miniforge {
    # If existing Miniforge python works, do NOT reinstall
    if (Test-WorkingPython $miniforgeExe) {
        Write-Host "Miniforge already present and working at $miniforgeBase - skipping install."
        return
    }

    # If directory exists but is not a valid install, install to a unique versioned folder
    $targetDir = $miniforgeBase
    if (Test-Path $targetDir) {
        $stamp = (Get-Date).ToString("yyyyMMdd-HHmmss")
        $targetDir = Join-Path $env:LOCALAPPDATA "miniforge3-$stamp"
        Write-Host "Existing folder '$miniforgeBase' is present but not usable. Installing to '$targetDir' instead."
    }

    Write-Host "Downloading Miniforge installer..."
    Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath

    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

    Write-Host "Installing Miniforge silently to $targetDir ..."
    # /S + /D=<path> (must be last, not quoted) is the documented conda-family silent install pattern [4](https://cefas.sharepoint.com/sites/PyFAS/Shared%20Documents/General/Mamba_Anaconda_Docs/20210722_PyFAS.pdf?web=1)[5](https://learn.microsoft.com/en-us/troubleshoot/windows-client/admin-development/create-desktop-shortcut-with-wsh)
    $args = @(
        "/S",
        "/InstallationType=JustMe",
        "/RegisterPython=0",
        "/AddToPath=0",
        "/D=$targetDir"
    )
    Start-Process -FilePath $installerPath -ArgumentList $args -Wait

    # If we installed to a versioned dir, use that as "current"
    $script:miniforgeBase = $targetDir
    $script:miniforgeExe  = Join-Path $targetDir "python.exe"

    if (!(Test-WorkingPython $script:miniforgeExe)) {
        throw "Miniforge python.exe not found or not runnable after install at $targetDir"
    }
}

function New-CleanVenv {
    param([string]$basePython, [string]$venvPath)

    if (Test-Path $venvPath) {
        Write-Host "Removing existing venv at $venvPath ..."
        Remove-Item -Recurse -Force $venvPath
    }

    Write-Host "Creating venv at $venvPath ..."
    & $basePython -m venv $venvPath | Out-Host

    $venvPy = Join-Path $venvPath "Scripts\python.exe"
    if (!(Test-Path $venvPy)) { throw "venv python not found at $venvPy" }

    # Force bootstrap pip even if the environment claims it exists
    Write-Host "Bootstrapping pip (ensurepip) ..."
    & $venvPy -m ensurepip --upgrade | Out-Host

    # Now upgrade pip via python -m pip (more robust than calling pip.exe)
    Write-Host "Upgrading pip..."
    & $venvPy -m pip install --upgrade pip | Out-Host

    # Quick sanity check: can we import pip?
    Write-Host "Verifying pip import..."
    & $venvPy -c "import pip; print(pip.__version__)" | Out-Host

    return $venvPy
}

function Install-Requirements {
    param([string]$venvPy, [string]$reqPath)

    Write-Host "Installing requirements..."
    & $venvPy -m pip install -r $reqPath | Out-Host
}

# ---------- Stage app files ----------
New-Item -ItemType Directory -Force -Path $here | Out-Null
#Copy-Item (Join-Path $here "requirements.txt")      -Destination (Join-Path $here "requirements.txt")      -Force
#Copy-Item (Join-Path $here "run_rapidplankton.bat") -Destination (Join-Path $here "run_rapidplankton.bat") -Force

# ---------- Ensure Miniforge ----------
Ensure-Miniforge

# ---------- Create venv + install deps (with retry if pip is broken) ----------
$req = Join-Path $here "requirements.txt"

try {
    $venvPy = New-CleanVenv -basePython $miniforgeExe -venvPath $venvDir
    Install-Requirements -venvPy $venvPy -reqPath $req
} catch {
    Write-Host "First attempt failed: $($_.Exception.Message)"
    Write-Host "Retrying once with a freshly recreated venv..."
    $venvPy = New-CleanVenv -basePython $miniforgeExe -venvPath $venvDir
    Install-Requirements -venvPy $venvPy -reqPath $req
}

# ---------- Create Desktop shortcut to the .bat launcher ----------
$desktop = [Environment]::GetFolderPath("Desktop")
$lnkPath = Join-Path $desktop "$appName.lnk"

$wsh = New-Object -ComObject WScript.Shell
$sc  = $wsh.CreateShortcut($lnkPath)

$sc.TargetPath       = "$env:WINDIR\System32\cmd.exe"
$sc.Arguments        = "/c `"$here\run_rapidplankton.bat`""
$sc.WorkingDirectory = $here
$sc.IconLocation     = "$env:SystemRoot\System32\shell32.dll,14"
$sc.Save()
