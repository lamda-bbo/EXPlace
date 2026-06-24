design=$1

make run_EXPlace DESIGN_CONFIG=designs/nangate45/${design}/config_EXPlace.mk
make run_wo_synth DESIGN_CONFIG=designs/nangate45/${design}/config_EXPlace.mk
