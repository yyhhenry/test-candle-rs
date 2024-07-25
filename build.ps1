if (Test-Path .\dist) { Remove-Item .\dist -Recurse -Force }

mkdir .\dist


cargo b -r -F cudnn --bin train --bin infer --bin bench
Copy-Item .\target\release\train.exe .\dist\train.exe
Copy-Item .\target\release\infer.exe .\dist\infer-gpu.exe
Copy-Item .\target\release\bench.exe .\dist\bench-gpu.exe

cargo b -r --bin infer --bin bench
Copy-Item .\target\release\infer.exe .\dist\infer-cpu.exe
Copy-Item .\target\release\bench.exe .\dist\bench-cpu.exe
