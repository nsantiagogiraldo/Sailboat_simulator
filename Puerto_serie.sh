#!/bin/bash
DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR/SNN_Codes/Spiking_codes && socat pty,link=interface_1,raw,echo=1 pty,link=interface_2,raw,echo=1

