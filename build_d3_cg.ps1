param([string]$GurobiRoot = 'C:\gurobi1300\win64')
$ErrorActionPreference = 'Stop'
$D3Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$D3Output = Join-Path $D3Root 'jt_oct\_native'
$D3VsWhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$D3Install = & $D3VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $D3Install) { throw 'MSVC C++ tools were not found' }
if (-not (Test-Path -LiteralPath (Join-Path $GurobiRoot 'include\gurobi_c++.h'))) { throw 'Gurobi 13 C++ SDK was not found' }
New-Item -ItemType Directory -Path $D3Output -Force | Out-Null
$D3VcVars = Join-Path $D3Install 'VC\Auxiliary\Build\vcvars64.bat'
$D3Command = '"' + $D3VcVars + '" >nul && cl /nologo /O2 /MD /openmp /EHsc /std:c++17 /LD /I"' + $GurobiRoot + '\include" /Fo"' + $D3Output + '\d3_cg.obj" "' + $D3Root + '\native\d3_cg.cpp" /link /LIBPATH:"' + $GurobiRoot + '\lib" gurobi_c++md2017.lib gurobi130.lib /OUT:"' + $D3Output + '\d3_cg.dll" /IMPLIB:"' + $D3Output + '\d3_cg.lib"'
cmd.exe /d /s /c $D3Command
if ($LASTEXITCODE -ne 0) { throw 'D3 C++/Gurobi build failed' }
Write-Output (Join-Path $D3Output 'd3_cg.dll')
