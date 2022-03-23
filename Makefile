d2_inst:
	-rm ML/custom_data.py
	ln -s custom_inst_data.py ML/custom_data.py

be_rd:
	-rm ML/custom_data.py
	ln -s custom_data_inout.py ML/custom_data.py
