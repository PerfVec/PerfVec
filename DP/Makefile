seq:
	g++ -O3 -std=c++11 0_buildSeq.cpp -o buildSeq
	g++ -O3 -std=c++11 -DUNIQUE_RD 0_buildSeq.cpp -o buildRD

dbg:
	g++ -O0 -g -std=c++11 0_buildSeq.cpp -o buildSeq
	g++ -O0 -g -std=c++11 0_buildInst.cpp -o buildInst

inst:
	g++ -O3 -std=c++11 0_buildInst.cpp -o buildInst
	g++ -O3 -std=c++11 -DUNIQUE_RD 0_buildInst.cpp -o buildInstRD

in:
	g++ -O3 -std=c++11 0_buildInstFeature.cpp -o buildInstFeature
	#g++ -g -O0 -std=c++11 0_buildInstFeature.cpp -o buildInstFeature

out:
	g++ -O3 -std=c++11 0_buildComOut.cpp -o buildComOut
