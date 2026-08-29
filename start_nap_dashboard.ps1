$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Port = 8522
$Url = "http://localhost:$Port"
$PreferredWorkbookName = "Nap_calendar_month_ultralight_formula.before-mopj-m3-fix-20260809-105658.xlsx"
$WorkbookSearchRoot = Join-Path $env:USERPROFILE "Nutstore\1"
$WorkbookPath = $null
$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$DashboardScript = Join-Path $ProjectDir "src\nap_dashboard.py"
$FallbackPython = "D:\Anacoda\python.exe"
$Python = $VenvPython
$OutputDir = Join-Path $ProjectDir "outputs"
$StdoutLog = Join-Path $OutputDir "nap_streamlit.out.log"
$StderrLog = Join-Path $OutputDir "nap_streamlit.err.log"
$LauncherLog = Join-Path $OutputDir "nap_launcher.log"

function Write-LauncherLog {
    param([string]$Message)
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $LauncherLog -Value "[$stamp] $Message" -Encoding UTF8
}

function Show-LauncherMessage {
    param([string]$Message)
    try {
        $shell = New-Object -ComObject WScript.Shell
        [void]$shell.Popup($Message, 8, "Reuters NAP Dashboard", 48)
    } catch {
        Write-LauncherLog $Message
    }
}

function Test-NapPort {
    try {
        return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    } catch {
        return $false
    }
}

function Get-NapPortProcess {
    try {
        $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $connection) {
            return $null
        }
        return Get-CimInstance Win32_Process -Filter "ProcessId = $($connection.OwningProcess)" -ErrorAction SilentlyContinue
    } catch {
        return $null
    }
}

try {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

    if ($env:NAP_DASHBOARD_WORKBOOK -and (Test-Path -LiteralPath $env:NAP_DASHBOARD_WORKBOOK)) {
        $WorkbookPath = (Resolve-Path -LiteralPath $env:NAP_DASHBOARD_WORKBOOK).Path
    } elseif (Test-Path -LiteralPath $WorkbookSearchRoot) {
        $preferredPattern = Join-Path $WorkbookSearchRoot "*\NAP-*\$PreferredWorkbookName"
        $WorkbookPath = Get-ChildItem -Path $preferredPattern -File -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1 -ExpandProperty FullName

        if (-not $WorkbookPath) {
            $fallbackPattern = Join-Path $WorkbookSearchRoot "*\NAP-*\Nap_calendar_month_ultralight_formula*.xlsx"
            $WorkbookPath = Get-ChildItem -Path $fallbackPattern -File -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1 -ExpandProperty FullName
        }
    }

    if (-not $WorkbookPath -or -not (Test-Path -LiteralPath $WorkbookPath)) {
        $message = "NAP source workbook was not found under:`n$WorkbookSearchRoot`nPreferred file: $PreferredWorkbookName"
        Write-LauncherLog $message
        Show-LauncherMessage $message
        exit 1
    }

    # nap_dashboard.py reads this value as the default workbook shown in the sidebar.
    # Start-Process inherits it, so the desktop shortcut always opens the audited source.
    $env:NAP_DASHBOARD_WORKBOOK = $WorkbookPath

    $existingProcess = Get-NapPortProcess
    if ($existingProcess) {
        $existingExecutable = [string]$existingProcess.ExecutablePath
        $existingCommand = [string]$existingProcess.CommandLine
        $isNapDashboard = $existingCommand -like "*nap_dashboard.py*"
        $usesProjectEnvironment = $existingCommand -like "*$DashboardScript*"
        if ($isNapDashboard -and -not $usesProjectEnvironment) {
            $message = "Port $Port is occupied by an older NAP Dashboard running outside the project environment.`nClose that dashboard process, then start again."
            Write-LauncherLog "$message Executable: $existingExecutable"
            Show-LauncherMessage $message
            exit 1
        }
        Write-LauncherLog "Port $Port is already listening with the expected runtime; opening $Url."
        Start-Process $Url
        exit 0
    }

    if (-not (Test-Path -LiteralPath $Python)) {
        $Python = $FallbackPython
    }

    if (-not (Test-Path -LiteralPath $Python)) {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd) {
            $Python = $cmd.Source
        } else {
            Show-LauncherMessage "Python was not found: $VenvPython or $FallbackPython`nPlease check the project environment."
            exit 1
        }
    }

    Write-LauncherLog "Starting Streamlit from $ProjectDir on port $Port using $Python -m streamlit. Workbook: $WorkbookPath"
    Start-Process `
        -FilePath $Python `
        -ArgumentList @("-m", "streamlit", "run", "$DashboardScript", "--server.port", "$Port", "--server.headless", "true", "--browser.gatherUsageStats", "false", "--server.fileWatcherType", "none", "--server.runOnSave", "false") `
        -WorkingDirectory $ProjectDir `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog `
        -WindowStyle Hidden

    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        if (Test-NapPort) {
            Start-Process $Url
            exit 0
        }
    }

    Show-LauncherMessage "Dashboard startup timed out. Check the log:`n$StderrLog"
    exit 1
} catch {
    Write-LauncherLog ("Launcher failed: " + $_.Exception.Message)
    Show-LauncherMessage ("Startup failed: " + $_.Exception.Message + "`nLog: " + $LauncherLog)
    exit 1
}
