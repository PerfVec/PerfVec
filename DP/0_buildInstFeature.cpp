#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>
#include <unordered_map>

using namespace std;

#include "inst.h"

Tick minCompleteLat = 100;
Tick minStoreLat = 100;

double minOut = 100;
double maxOut = 0;

#include "inst_noflush_impl.h"

int main(int argc, char *argv[]) {
  assert(IN_LEN == 51 && IN_REG_SRC_BEGIN == 23 &&
         IN_REG_DST_BEGIN == 39 && IN_OP == 1);
  if (argc != 2) {
    cerr << "Usage: ./buildInstFeature <trace>" << endl;
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
  outputName.replace(outputName.end()-3, outputName.end(), "in");
  cerr << "Write to " << outputName << ".\n";
  ofstream output(outputName);
  if (!output.is_open()) {
    cerr << "Cannot open output file.\n";
    return 0;
  }

  double buf[IN_LEN];

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
  BranchHistories.clear();
#ifdef UNIQUE_RD
  pcArr.clear();
  dataLdArr.clear();
  dataStArr.clear();
#elif defined(TREE_RD)
  PCRD = new ReuseDistance();
  LdRD = new ReuseDistance();
  StRD = new ReuseDistance();
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
      inst.dumpFeatures(curTick, &buf[0]);
      // Dump instruction.
      for (int j = 0; j < IN_LEN; j++) {
        output << buf[j] << " ";
        if (buf[j] > maxOut)
          maxOut = buf[j];
        if (buf[j] < minOut)
          minOut = buf[j];
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
#if defined(TREE_RD)
  delete PCRD;
  delete LdRD;
  delete StRD;
#endif
  return 0;
}
