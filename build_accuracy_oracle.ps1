param([string]$CudaPackages = '')
$ErrorActionPreference = 'Stop'
$OracleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $CudaPackages) { $CudaPackages = Join-Path $OracleRoot '.venv\Lib\site-packages\nvidia' }
if (-not (Test-Path -LiteralPath (Join-Path $CudaPackages 'cuda_runtime\include\cuda.h'))) { throw 'Pass -CudaPackages with the installed nvidia Python package directory' }
$OracleOutput = Join-Path $OracleRoot 'jt_oct\_native'
$OracleInstall = & 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe' -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $OracleInstall) { throw 'MSVC C++ tools were not found' }
$OracleVcVars = Join-Path $OracleInstall 'VC\Auxiliary\Build\vcvars64.bat'
$OracleCommand = '"' + $OracleVcVars + '" >nul && cl /nologo /O2 /MD /EHsc /std:c++17 /openmp /LD /I"' + $CudaPackages + '\cuda_runtime\include" /I"' + $CudaPackages + '\cuda_nvrtc\include" /Fo"' + $OracleOutput + '\accuracy_oracle.obj" "' + $OracleRoot + '\native\accuracy_oracle.cpp" /link /LIBPATH:"' + $CudaPackages + '\cuda_runtime\lib\x64" cuda.lib psapi.lib /OUT:"' + $OracleOutput + '\accuracy_oracle.dll" /IMPLIB:"' + $OracleOutput + '\accuracy_oracle.lib"'
cmd.exe /d /s /c $OracleCommand
if ($LASTEXITCODE -ne 0) { throw 'Accuracy oracle build failed' }
