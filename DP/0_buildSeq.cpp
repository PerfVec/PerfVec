#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>
#include <string>
#include <unordered_map>

using namespace std;

#include "inst.h"

// Current sequence status.
int curInstNum;
Tick instIdx;
Tick memLdIdx;
Tick memStIdx;
Tick lastFetchTick;
Tick lastCommitTick;
Tick lastSqOutTick;
unordered_map<Addr, int> pcMap;
unordered_map<Addr, Tick> pcLMap;
unordered_map<Addr, int> dataMap;
unordered_map<Addr, int> dataLineMap;
unordered_map<Addr, Tick> dataLineLdMap;
unordered_map<Addr, Tick> dataLineStMap;

#define TICK_STEP 500
Tick minCompleteLat = 100;
Tick minStoreLat = 100;

int minOut = 100;
int maxOut = 0;

#include "inst_impl.h"

int main(int argc, char *argv[]) {
  assert(TRAIN_INST_LEN == 61 && IN_REG_SRC_BEGIN == 33 &&
         IN_REG_DST_BEGIN == 49 && IN_OP == 12 && IN_ST == 13);
  if (argc != 2) {
    cerr << "Usage: ./buildSeq <trace>" << endl;
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
  outputName.replace(outputName.end()-3, outputName.end(), "seq");
  cerr << "Write to " << outputName << ".\n";
  ofstream output(outputName);
  if (!output.is_open()) {
    cerr << "Cannot open output file.\n";
    return 0;
  }

  cerr << "Training data are " << SEQ_LEN << " by " << TRAIN_INST_LEN << ".\n";
  int *seq = new int[SEQ_LEN * TRAIN_INST_LEN];
  int buf[TRAIN_INST_LEN];

  // First fetch starts at the 2nd cycle.
  Tick curTick = 2;
  bool firstInst = true;
  Tick num = 0;
  Tick discardNum = 0;
  int seqNum = 0;
  Inst inst;

  curInstNum = 0;
  instIdx = 0;
  memLdIdx = 0;
  memStIdx = 0;
  lastFetchTick = 0;
  lastCommitTick = 0;
  lastSqOutTick = 0;
  pcMap.clear();
  pcLMap.clear();
  dataMap.clear();
  dataLineMap.clear();
  dataLineLdMap.clear();
  dataLineStMap.clear();
  while (!trace.eof()) {
    Tick res = inst.read(trace, sqtrace);
    if (res == FILE_END)
      break;
    else if (res == READ_INST) {
      if (curInstNum < SEQ_LEN) {
        inst.dump(curTick, &seq[curInstNum * TRAIN_INST_LEN]);
        curInstNum++;
        num++;
        if (num % 100000 == 0)
          cerr << ".";
        if (num % 10000000 == 0)
          cerr << "\n";
      } else {
        // Update memory access maps etc.
        inst.dump(curTick, &buf[0]);
        discardNum++;
      }
      instIdx++;
      if (inst.isLoad())
        memLdIdx++;
      else if (inst.isStore())
        memStIdx++;
    } else {
      assert(curInstNum == SEQ_LEN);
      seqNum++;
      // Fetch starts after 3 cycles.
      curTick = res + 3;
      curInstNum = 0;
      lastFetchTick = 0;
      lastCommitTick = 0;
      lastSqOutTick = 0;
      pcMap.clear();
      dataMap.clear();
      dataLineMap.clear();
      // Dump sequence.
      for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < TRAIN_INST_LEN; j++) {
          output << seq[i * TRAIN_INST_LEN + j] << " ";
          if (seq[i * TRAIN_INST_LEN + j] > maxOut)
            maxOut = seq[i * TRAIN_INST_LEN + j];
          if (seq[i * TRAIN_INST_LEN + j] < minOut)
            minOut = seq[i * TRAIN_INST_LEN + j];
        }
        output << "\n";
      }
    }
  }

  cerr << "\nFinish with " << num << " instructions (discard " << discardNum
       << " instructions) and " << seqNum << " sequences.\n";
  cerr << "Min complete and store latency is " << minCompleteLat << " "
       << minStoreLat << ".\n";
  cerr << "Min and max output value is " << minOut << " " << maxOut << ".\n";
  trace.close();
  sqtrace.close();
  output.close();
  delete seq;
  return 0;
}
