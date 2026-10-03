# Puts a "Dub Studio" icon on the desktop that opens the studio in its own window.
#   powershell -ExecutionPolicy Bypass -File make_shortcut.ps1 [-Env dub]
param([string]$Env = "dub")

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = (conda run -n $Env python -c "import sys; print(sys.executable)" | Where-Object { $_ -match "python\.exe$" } | Select-Object -Last 1)
if (-not $python) { throw "conda env '$Env' not found" }
$python = $python.Trim()
$pythonw = Join-Path (Split-Path $python) "pythonw.exe"
if (-not (Test-Path $pythonw)) { throw "pythonw.exe not found next to $python" }

# icon from the watermark logo (wide, so centred on a transparent square)
$ico = Join-Path $root "studio\static\dub-studio.ico"
& $python -c "from PIL import Image; im = Image.open(r'$root\watermark-nobackground.png').convert('RGBA'); n = max(im.size); sq = Image.new('RGBA', (n, n)); sq.paste(im, ((n - im.width) // 2, (n - im.height) // 2)); sq.save(r'$ico', sizes=[(16,16),(32,32),(48,48),(256,256)])"

$lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "Dub Studio.lnk"
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
$s.TargetPath = $pythonw
$s.Arguments = "-m studio.desktop"
$s.WorkingDirectory = $root
$s.IconLocation = $ico
$s.Description = "Dub Studio"
$s.Save()
Write-Host "Created $lnk"
