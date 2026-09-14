$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Source = Join-Path $ProjectRoot "native\d3_kappa.cpp"
$OutputDir = Join-Path $ProjectRoot "jt_oct\_native"
$Output = Join-Path $OutputDir "d3_kappa.dll"
$Object = Join-Path $OutputDir "d3_kappa.obj"
$ImportLibrary = Join-Path $OutputDir "d3_kappa.lib"
$VsWhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $VsWhere)) {
    throw "Visual Studio Build Tools were not found"
}
$Install = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $Install) {
    throw "MSVC C++ tools were not found"
}
$VcVars = Join-Path $Install "VC\Auxiliary\Build\vcvars64.bat"
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$Command = '"' + $VcVars + '" >nul && cl /nologo /O2 /openmp /EHsc /std:c++17 /LD /Fo"' + $Object + '" "' + $Source + '" /link /OUT:"' + $Output + '" /IMPLIB:"' + $ImportLibrary + '"'
cmd.exe /d /s /c $Command
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $Output)) {
    throw "Native backend build failed"
}
Write-Output $Output
