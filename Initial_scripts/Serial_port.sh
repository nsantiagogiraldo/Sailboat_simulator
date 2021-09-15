#!/bin/bash
cd Documentos/Ubuntu_Maestria/SNN_Codes/Canal-Puertos
socat pty,link=interface_1,raw,echo=1 pty,link=interface_2,raw,echo=1
