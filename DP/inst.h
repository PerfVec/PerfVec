#ifndef __INST_H__
#define __INST_H__

#define READ_INST 0
#define FILE_END 1

#define SRCREGNUM 8
#define DSTREGNUM 6
#define MAXREGCLASS 6
#define MAXREGIDX 50

typedef long unsigned Tick;
typedef long unsigned Addr;

struct Inst {
  // Operation.
  int isFault;
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
  int isRdBar;
  int isWrBar;
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
  long sqIdx;

  // Instruction access.
  Addr pc;
  int isBranching;
  int isMisPredict;
  int fetchDepth;
  int iwalkDepth[3];
  Addr iwalkAddr[3];
  int iWritebacks[2];

  // Timing.
  Tick inTick;
  Tick completeTick;
  Tick outTick;
  Tick decodeTick;
  Tick renameTick;
  Tick dispatchTick;
  Tick issueTick;
  Tick storeTick;
  Tick sqOutTick;

  // Read one instruction from SQ and ROB traces.
  Tick read(ifstream &ROBtrace, ifstream &SQtrace, bool isSingleTrace = false);

  // Whether it is a normal store.
  bool inSQ() {
    if (sqIdx != -1 && !isStoreConditional && !isAtomic)
      return true;
    else
      return false;
  }

  bool isLoad() { return (op == 47); }
  bool isStore() { return (op == 48); }

  void dump(Tick startTick, double *out);
  void dumpTargets(Tick startTick, double *out);
  void dumpTargets(Tick startTick, double *out, Tick &memLdIdx, Tick &memStIdx, Tick &lastFetchTick, Tick &lastCommitTick, Tick &lastSqOutTick, Tick &lastDecodeTick, Tick &lastRenameTick, Tick &lastDispatchTick);
  void dumpFeatures(Tick startTick, double *out);
};

#endif
