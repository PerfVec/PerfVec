#ifndef __INST_H__
#define __INST_H__

#define READ_INST 0
#define FILE_END 1

#define SRCREGNUM 8
#define DSTREGNUM 6
#define MAXREGCLASS 6
#define MAXREGIDX 50

#define SEQ_LEN 1024
#define TRAIN_INST_LEN (24 + 2*SRCREGNUM + 2*DSTREGNUM)

typedef long unsigned Tick;
typedef long unsigned Addr;

struct Inst {
  // Operation.
  int op;
  int isMicroOp;
  int isCondCtrl;
  int isUncondCtrl;
  int isDirectCtrl;
  int isSquashAfter;
  int isSerializeAfter;
  int isSerializeBefore;
  int isAtomic;
  int isStoreConditional;
  int isMemBar;
  int isQuiesce;
  int isNonSpeculative;

  // Registers.
  int srcNum;
  int destNum;
  int srcClass[SRCREGNUM];
  int srcIndex[SRCREGNUM];
  int destClass[DSTREGNUM];
  int destIndex[DSTREGNUM];

  // Data access.
  int isAddr;
  Addr addr;
  Addr addrEnd;
  unsigned int size;
  int depth;
  int dwalkDepth[3];
  Addr dwalkAddr[3];
  int dWritebacks[3];
  int sqIdx;

  // Instruction access.
  Addr pc;
  int isMisPredict;
  int fetchDepth;
  int iwalkDepth[3];
  Addr iwalkAddr[3];
  int iWritebacks[2];

  // Timing.
  Tick inTick;
  Tick completeTick;
  Tick outTick;
  Tick storeTick;
  Tick sqOutTick;

  // Read one instruction from SQ and ROB traces.
  Tick read(ifstream &ROBtrace, ifstream &SQtrace);

  // Whether it is a normal store.
  bool inSQ() {
    if (sqIdx != -1 && !isStoreConditional && !isAtomic)
      return true;
    else
      return false;
  }

  // Get ticks.
  Tick robTick() { return inTick + outTick; }
  Tick sqTick() {
    if (sqOutTick == 0)
      return inTick + storeTick;
    else
      return inTick + sqOutTick;
  }

  void dump(Tick startTick, int *out);
};

#endif
