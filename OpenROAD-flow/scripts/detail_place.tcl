utl::set_metrics_stage "detailedplace__{}"
source $::env(SCRIPTS_DIR)/load.tcl
erase_non_stage_variables place
load_design 3_4_place_resized.odb 2_floorplan.sdc

source $::env(PLATFORM_DIR)/setRC.tcl

proc do_dpl {} {
  # Only for use with hybrid rows
  if {[env_var_equals BALANCE_ROWS 1]} {
    balance_row_usage
  }
  
  set_placement_padding -global \
      -left $::env(CELL_PAD_IN_SITES_DETAIL_PLACEMENT) \
      -right $::env(CELL_PAD_IN_SITES_DETAIL_PLACEMENT)

  if {[env_var_exists_and_non_empty LG_MAX_DISPLACEMENT]} {
      log_cmd detailed_placement -max_displacement $::env(LG_MAX_DISPLACEMENT)
  } else {
    log_cmd detailed_placement
  }

  if {[env_var_equals ENABLE_DPO 1]} {
    if {[env_var_exists_and_non_empty DPO_MAX_DISPLACEMENT]} {
      log_cmd improve_placement -max_displacement $::env(DPO_MAX_DISPLACEMENT)
    } else {
      log_cmd improve_placement
    }
  }
  log_cmd optimize_mirroring

  if {[env_var_exists_and_non_empty SKIP_DP_CHECK]} {
    puts "Skip detailed placement check for RePlAce"
  }  else {
    utl::info FLW 12 "Placement violations [check_placement -verbose]."
  }
  
  
  estimate_parasitics -placement
}

set result [catch {do_dpl} errMsg]
if {$result != 0} {
  write_db $::env(RESULTS_DIR)/3_5_place_dp-failed.odb
  # error $errMsg
}

if {![env_var_exists_and_non_empty SKIP_ALL_REPORTS]} {
  report_metrics 3 "detailed place" true false
}

write_db $::env(RESULTS_DIR)/3_5_place_dp.odb
