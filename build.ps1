if (Test-Path .\dist) { Remove-Item .\dist -Recurse -Force }

mkdir .\dist


cargo b -r -F cudnn --bin train
cargo b -r

Copy-Item .\target\release\*.exe .\dist\

cargo b -r -F cuda --bin infer

Copy-Item .\target\release\infer.exe .\dist\infer-cuda.exe
