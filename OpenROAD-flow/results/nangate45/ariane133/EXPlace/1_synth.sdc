set sdc_version 2.0

# Set the current design
current_design ariane

create_clock -name "core_clock" -period 3.0 -waveform {0.0 1.5} [get_ports clk_i]
