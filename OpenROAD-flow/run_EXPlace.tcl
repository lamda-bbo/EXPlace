read_lef $::env(TECH_LEF)
read_lef $::env(SC_LEF)
if {[info exist ::env(ADDITIONAL_LEFS)]} {
foreach lef $::env(ADDITIONAL_LEFS) {
    read_lef $lef
}
}
foreach lib_file $env(LIB_FILES) {
  read_lib $lib_file
}

read_def $::env(MACRO_DEF).gp.def
write_macro_placement $::env(RESULTS_DIR)/macro_out