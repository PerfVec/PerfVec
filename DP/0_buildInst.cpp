#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>
#include <unordered_map>

using namespace std;

#include "inst.h"

//#define UNIQUE_RD

// Current sequence status.
Tick instIdx;
Tick memLdIdx;
Tick memStIdx;
Tick lastFetchTick;
Tick lastCommitTick;
Tick lastSqOutTick;
Tick lastDecodeTick;
Tick lastRenameTick;
Tick lastDispatchTick;
unordered_map<Addr, Tick> dataMap;
#ifdef UNIQUE_RD
vector<Addr> pcArr;
vector<Addr> dataLdArr;
vector<Addr> dataStArr;
#else
unordered_map<Addr, Tick> pcLMap;
unordered_map<Addr, Tick> dataLineLdMap;
unordered_map<Addr, Tick> dataLineStMap;
#endif

#define TICK_STEP 500
Tick minCompleteLat = 100;
Tick minStoreLat = 100;

int minOut = 100;
int maxOut = 0;

#include "inst_noflush_impl.h"

int main(int argc, char *argv[]) {
  assert(TRAIN_INST_LEN == 63 && IN_REG_SRC_BEGIN == 35 &&
         IN_REG_DST_BEGIN == 51 && IN_OP == 15 && IN_ST == 16);
  if (argc != 2) {
    cerr << "Usage: ./buildInst <trace>" << endl;
    return 0;
  }
  ifstream trace(argv[1]);
  if (!trace.is_open()) {
    cerr << "Cannot open trace file.\n";
    return 0;
  }
  string outputName = argv[1];
  outputName.replace(outputName.end()-3, outputName.end(), "sq.txt");
  ifstream sqtrace(outputName);
  if (!sqtrace.is_open()) {
    cerr << "Cannot open SQ trace file " << outputName << ".\n";
    return 0;
  }

  outputName = argv[1];
  outputName.replace(outputName.end()-3, outputName.end(), "inst");
  cerr << "Write to " << outputName << ".\n";
  ofstream output(outputName);
  if (!output.is_open()) {
    cerr << "Cannot open output file.\n";
    return 0;
  }

  int *seq = new int[TRAIN_INST_LEN];
  int buf[TRAIN_INST_LEN];

  // First fetch starts at cycle 0.
  Tick curTick = 0;
  bool firstInst = true;
  Inst inst;

  instIdx = 0;
  memLdIdx = 0;
  memStIdx = 0;
  lastFetchTick = 0;
  lastCommitTick = 0;
  lastSqOutTick = 0;
  lastDecodeTick = 0;
  lastRenameTick = 0;
  lastDispatchTick = 0;
  dataMap.clear();
#ifdef UNIQUE_RD
  pcArr.clear();
  dataLdArr.clear();
  dataStArr.clear();
#else
  pcLMap.clear();
  dataLineLdMap.clear();
  dataLineStMap.clear();
#endif
  while (!trace.eof()) {
    Tick res = inst.read(trace, sqtrace);
    if (res == FILE_END)
      break;
    else if (res == READ_INST) {
      if (firstInst) {
        inst.fetchDepth = 2;
        firstInst = false;
      }
      inst.dump(curTick, &seq[0]);
      // Dump instruction.
      for (int j = 0; j < TRAIN_INST_LEN; j++) {
        output << seq[j] << " ";
        if (seq[j] > maxOut)
          maxOut = seq[j];
        if (seq[j] < minOut)
          minOut = seq[j];
      }
      output << "\n";
      instIdx++;
      if (inst.isLoad())
        memLdIdx++;
      else if (inst.isStore())
        memStIdx++;
      if (instIdx % 100000 == 0)
        cerr << ".";
      if (instIdx % 10000000 == 0)
        cerr << "\n";
    } else {
      assert(0);
    }
  }

  cerr << "\nFinish with " << instIdx << " instructions.\n";
  cerr << "Min complete and store latency is " << minCompleteLat << " "
       << minStoreLat << ".\n";
  cerr << "Min and max output value is " << minOut << " " << maxOut << ".\n";
  trace.close();
  sqtrace.close();
  output.close();
  delete seq;
  return 0;
}
