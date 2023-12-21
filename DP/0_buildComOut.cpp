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

int minOut = 100;
int maxOut = 0;

#include "inst_noflush_impl.h"

class SimModule {
public:
  bool isSingleTrace;
  ifstream trace;
  ifstream sqtrace;

  // First fetch starts at cycle 0.
  Tick curTick = 0;
  Tick memLdIdx = 0;
  Tick memStIdx = 0;
  Tick lastFetchTick = 0;
  Tick lastCommitTick = 0;
  Tick lastSqOutTick = 0;
  Tick lastDecodeTick = 0;
  Tick lastRenameTick = 0;
  Tick lastDispatchTick = 0;

  ~SimModule() {
    trace.close();
    sqtrace.close();
  }

  bool init(const char *name) {
    trace.open(name);
    if (!trace.is_open()) {
      cerr << "Cannot open trace file " << name << ".\n";
      return false;
    }
    string sqName = name;
    sqName.replace(sqName.end()-3, sqName.end(), "sq.txt");
    sqtrace.open(sqName);
    if (!sqtrace.is_open()) {
      cerr << "Single trace input (" << sqName << ").\n";
      isSingleTrace = true;
    } else
      isSingleTrace = false;
    return true;
  }

  Tick readInst(Inst &inst) {
    return inst.read(trace, sqtrace, isSingleTrace);
  }

  void update(Inst &inst, double *out) {
    inst.dumpTargets(curTick, out, memLdIdx, memStIdx, lastFetchTick, lastCommitTick, lastSqOutTick, lastDecodeTick, lastRenameTick, lastDispatchTick);
    if (inst.isLoad())
      memLdIdx++;
    else if (inst.isStore())
      memStIdx++;
  }
};

int main(int argc, char *argv[]) {
  assert(TGT_LEN == 16);
  if (argc < 2) {
    cerr << "Usage: ./buildComOut <trace> ..." << endl;
    return 0;
  }
  int file_num = argc - 1;
  SimModule *modules = new SimModule[file_num];
  for (int i = 1; i < argc; i++)
    if (!modules[i - 1].init(argv[i]))
      return 0;

  string outputName = argv[1];
  outputName.replace(outputName.end()-3, outputName.end(), "out");
  cerr << "Write to " << outputName << ".\n";
  ofstream output(outputName);
  if (!output.is_open()) {
    cerr << "Cannot open target output file.\n";
    return 0;
  }

  double *seq = new double[TGT_LEN * file_num];

  bool firstInst = true;
  Inst *insts = new Inst[file_num];

  while (true) {
    int i;
    for (i = 0; i < file_num; i++) {
      Tick res = modules[i].readInst(insts[i]);
      if (res == FILE_END)
        break;
      assert(res == READ_INST);
      if (insts[i].pc != insts[0].pc ||
          insts[i].isBranching != insts[0].isBranching) {
        cerr << "\nEarly stop at instruction " << instIdx << " (" << i << " "
             << insts[i].pc << " " << insts[0].pc << " " << insts[i].isBranching
             << " " << insts[0].isBranching << ")\n";
        printOP(&insts[0]);
        break;
      }
      if (firstInst)
        insts[i].fetchDepth = 2;
    }
    if (i < file_num)
      break;
    firstInst = false;
    for (i = 0; i < file_num; i++)
      modules[i].update(insts[i], &seq[i * TGT_LEN]);
    // Dump instruction.
    for (int j = 0; j < TGT_LEN * file_num; j++) {
      output << seq[j] << " ";
      if (seq[j] > maxOut)
        maxOut = seq[j];
      if (seq[j] < minOut)
        minOut = seq[j];
    }
    output << "\n";
    instIdx++;
    if (instIdx % 100000 == 0)
      cerr << ".";
    if (instIdx % 10000000 == 0)
      cerr << "\n";
  }

  cerr << "\nFinish with " << instIdx << " instructions.\n";
  cerr << "Min complete and store latency is " << minCompleteLat << " "
       << minStoreLat << ".\n";
  cerr << "Min and max output value is " << minOut << " " << maxOut << ".\n";
  delete[] modules;
  output.close();
  delete seq;
  return 0;
}
