param([string]$GurobiRoot = 'C:\gurobi1300\win64')
$ErrorActionPreference = 'Stop'
$CrRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$CrOutput = Join-Path $CrRoot 'jt_oct\_native'
$CrVsWhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$CrInstall = & $CrVsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $CrInstall) { throw 'MSVC C++ tools were not found' }
New-Item -ItemType Directory -Path $CrOutput -Force | Out-Null
$CrVcVars = Join-Path $CrInstall 'VC\Auxiliary\Build\vcvars64.bat'
$CrCommand = '"' + $CrVcVars + '" >nul && cl /nologo /O2 /MD /EHsc /std:c++17 /LD /I"' + $GurobiRoot + '\include" /Fo"' + $CrOutput + '\sparse_tail_rmp.obj" "' + $CrRoot + '\native\sparse_tail_rmp.cpp" /link /LIBPATH:"' + $GurobiRoot + '\lib" gurobi_c++md2017.lib gurobi130.lib /OUT:"' + $CrOutput + '\sparse_tail_rmp.dll" /IMPLIB:"' + $CrOutput + '\sparse_tail_rmp.lib"'
cmd.exe /d /s /c $CrCommand
if ($LASTEXITCODE -ne 0) { throw 'Contracted Gurobi RMP build failed' }

