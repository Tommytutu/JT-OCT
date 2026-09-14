$ErrorActionPreference = "Stop"
$D3Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$D3Source = Join-Path $D3Root "native\d3_optimized.cpp"
$D3OutputDir = Join-Path $D3Root "jt_oct\_native"
$D3Output = Join-Path $D3OutputDir "d3_optimized.dll"
$D3Object = Join-Path $D3OutputDir "d3_optimized.obj"
$D3Import = Join-Path $D3OutputDir "d3_optimized.lib"
$D3VsWhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
$D3Install = & $D3VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $D3Install) { throw "MSVC C++ tools were not found" }
$D3VcVars = Join-Path $D3Install "VC\Auxiliary\Build\vcvars64.bat"
$D3Command = '"' + $D3VcVars + '" >nul && cl /nologo /O2 /openmp /EHsc /std:c++17 /LD /Fo"' + $D3Object + '" "' + $D3Source + '" /link /OUT:"' + $D3Output + '" /IMPLIB:"' + $D3Import + '"'
cmd.exe /d /s /c $D3Command
if ($LASTEXITCODE -ne 0) { throw "D3 optimized native backend build failed" }
Write-Output $D3Output
