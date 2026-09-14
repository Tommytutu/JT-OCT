param([string]$GurobiRoot = 'C:\gurobi1300\win64')
$ErrorActionPreference = 'Stop'
$D2Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$D2Output = Join-Path $D2Root 'jt_oct\_native'
$D2VsWhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$D2Install = & $D2VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $D2Install) { throw 'MSVC C++ tools were not found' }
if (-not (Test-Path -LiteralPath (Join-Path $GurobiRoot 'include\gurobi_c++.h'))) { throw 'Gurobi 13 C++ SDK was not found' }
New-Item -ItemType Directory -Path $D2Output -Force | Out-Null
$D2VcVars = Join-Path $D2Install 'VC\Auxiliary\Build\vcvars64.bat'
$D2Command = '"' + $D2VcVars + '" >nul && cl /nologo /O2 /MD /openmp /EHsc /std:c++17 /LD /I"' + $GurobiRoot + '\include" /Fo"' + $D2Output + '\d2_cg.obj" "' + $D2Root + '\native\d2_cg.cpp" /link /LIBPATH:"' + $GurobiRoot + '\lib" gurobi_c++md2017.lib gurobi130.lib /OUT:"' + $D2Output + '\d2_cg.dll" /IMPLIB:"' + $D2Output + '\d2_cg.lib"'
cmd.exe /d /s /c $D2Command
if ($LASTEXITCODE -ne 0) { throw 'D2 C++/Gurobi build failed' }
Write-Output (Join-Path $D2Output 'd2_cg.dll')
