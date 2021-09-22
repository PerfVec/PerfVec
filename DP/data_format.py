# targets.
tgt_fetch_lat = 0
tgt_completion_lat = 1
tgt_commit_lat = 2
tgt_store_lat = 3
tgt_store_commit_lat = 4

# FIXME
# instruction fields.
inst_op               = 0
inst_insq             = inst_op + 1
inst_micro            = inst_insq + 1
inst_mispred          = inst_micro + 1
inst_cctrl            = inst_mispred + 1
inst_ucctrl           = inst_cctrl + 1
inst_dctrl            = inst_ucctrl + 1
inst_squash_af        = inst_dctrl + 1
inst_serial_af        = inst_squash_af + 1
inst_serial_be        = inst_serial_af + 1
#inst_atom             = inst_serial_be + 1
inst_storec           = inst_serial_be + 1
inst_membar           = inst_storec + 1
#inst_quiesce          = inst_membar + 1
inst_nonspeculative   = inst_membar + 1

inst_srcreg_begin     = inst_nonspeculative + 1
inst_srcreg_end       = inst_srcreg_begin + 7
inst_dstreg_begin     = inst_srcreg_end + 1
inst_dstreg_end       = inst_dstreg_begin + 5

inst_fetch_depth      = inst_dstreg_end + 1
inst_fetch_linec      = inst_fetch_depth + 1
#inst_fetch_walk_begin = inst_fetch_linec + 1
#inst_fetch_walk_end   = inst_fetch_walk_begin + 2
#inst_fetch_pagec      = inst_fetch_walk_end + 1
#inst_fetch_wb_begin   = inst_fetch_pagec + 1
#inst_fetch_wb_end     = inst_fetch_wb_begin + 1

inst_data_depth       = inst_fetch_linec + 1
inst_data_addrc       = inst_data_depth + 1
inst_data_linec       = inst_data_addrc + 1
#inst_data_walk_begin  = inst_data_linec + 1
#inst_data_walk_end    = inst_data_walk_begin + 2
#inst_data_pagec       = inst_data_walk_end + 1
#inst_data_wb_begin    = inst_data_pagec + 1
#inst_data_wb_end      = inst_data_wb_begin + 2
#assert inst_data_linec == inst_length - 1

# trace format.
tr_sq_offset = 2
tr_sqidx                  = 0
tr_fetch_cycle            = tr_sqidx + 1
tr_complete_lat           = tr_fetch_cycle + 1
tr_commit_lat             = tr_complete_lat + 1
tr_store_lat              = tr_commit_lat + 1
tr_store_commit_lat       = tr_store_lat + 1

tr_op                     = tr_commit_lat + 1
tr_micro                  = tr_op + 1
tr_cctrl                  = tr_micro + 1
tr_ucctrl                 = tr_cctrl + 1
tr_dctrl                  = tr_ucctrl + 1
tr_squash_af              = tr_dctrl + 1
tr_serial_af              = tr_squash_af + 1
tr_serial_be              = tr_serial_af + 1
tr_atom                   = tr_serial_be + 1
tr_storec                 = tr_atom + 1
tr_membar                 = tr_storec + 1
tr_quiesce                = tr_membar + 1
tr_nonspeculative         = tr_quiesce + 1

tr_data_valid             = tr_nonspeculative + 1
tr_data_addr              = tr_data_valid + 1
tr_data_size              = tr_data_addr + 1
tr_data_depth             = tr_data_size + 1
tr_data_walk_depth_begin  = tr_data_depth + 1
tr_data_walk_depth_end    = tr_data_walk_depth_begin + 2
tr_data_walk_addr_begin   = tr_data_walk_depth_end + 1
tr_data_walk_addr_end     = tr_data_walk_addr_begin + 2
tr_data_wb_begin          = tr_data_walk_addr_end + 1
tr_data_wb_end            = tr_data_wb_begin + 2

tr_fetch_pc               = tr_data_wb_end + 1
tr_fetch_mispred          = tr_fetch_pc + 1
tr_fetch_depth            = tr_fetch_mispred + 1
tr_fetch_walk_depth_begin = tr_fetch_depth + 1
tr_fetch_walk_depth_end   = tr_fetch_walk_depth_begin + 2
tr_fetch_walk_addr_begin  = tr_fetch_walk_depth_end + 1
tr_fetch_walk_addr_end    = tr_fetch_walk_addr_begin + 2
tr_fetch_wb_begin         = tr_fetch_walk_addr_end + 1
tr_fetch_wb_end           = tr_fetch_wb_begin + 1

tr_src_num                = tr_fetch_wb_end + 1
tr_dst_num                = tr_src_num + 1
