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

  bool isLoad() { return (op == 52 || op == 54 || op == 58 || op == 60 || op == 62 || op == 64 || op == 66 || op == 68 || op == 69); }
  bool isStore() { return (op == 53 || op == 55 || op == 57 || op == 59 || op == 61 || op == 63 || op == 65 || op == 67 || op == 70); }
  

  void dump(Tick startTick, double *out);
  void dumpTargets(Tick startTick, double *out);
  void dumpTargets(Tick startTick, double *out, Tick &memLdIdx, Tick &memStIdx, Tick &lastFetchTick, Tick &lastCommitTick, Tick &lastSqOutTick, Tick &lastDecodeTick, Tick &lastRenameTick, Tick &lastDispatchTick);
  void dumpFeatures(Tick startTick, double *out);
  
  void print() const {
        cout << "Operation:" << endl;
        cout << "  isFault: " << isFault << endl;
        cout << "  op: " << op << endl;
        cout << "  isMicroOp: " << isMicroOp << endl;
        cout << "  isCondCtrl: " << isCondCtrl << endl;
        cout << "  isUncondCtrl: " << isUncondCtrl << endl;
        cout << "  isDirectCtrl: " << isDirectCtrl << endl;
        cout << "  isSquashAfter: " << isSquashAfter << endl;
        cout << "  isSerializeAfter: " << isSerializeAfter << endl;
        cout << "  isSerializeBefore: " << isSerializeBefore << endl;
        cout << "  isAtomic: " << isAtomic << endl;
        cout << "  isStoreConditional: " << isStoreConditional << endl;
        cout << "  isMemBar: " << isMemBar << endl;
        cout << "  isRdBar: " << isRdBar << endl;
        cout << "  isWrBar: " << isWrBar << endl;
        cout << "  isQuiesce: " << isQuiesce << endl;
        cout << "  isNonSpeculative: " << isNonSpeculative << endl;

        cout << "Registers:" << endl;
        cout << "  srcNum: " << srcNum << endl;
        cout << "  destNum: " << destNum << endl;
        cout << "  srcClass: ";
        for (int i = 0; i < SRCREGNUM; ++i) cout << srcClass[i] << " ";
        cout << endl;
        cout << "  srcIndex: ";
        for (int i = 0; i < SRCREGNUM; ++i) cout << srcIndex[i] << " ";
        cout << endl;
        cout << "  destClass: ";
        for (int i = 0; i < DSTREGNUM; ++i) cout << destClass[i] << " ";
        cout << endl;
        cout << "  destIndex: ";
        for (int i = 0; i < DSTREGNUM; ++i) cout << destIndex[i] << " ";
        cout << endl;

        cout << "Data access:" << endl;
        cout << "  isAddr: " << isAddr << endl;
        cout << "  addr: " << addr << endl;
        cout << "  addrEnd: " << addrEnd << endl;
        cout << "  size: " << size << endl;
        cout << "  depth: " << depth << endl;
        cout << "  dwalkDepth: ";
        for (int i = 0; i < 3; ++i) cout << dwalkDepth[i] << " ";
        cout << endl;
        cout << "  dwalkAddr: ";
        for (int i = 0; i < 3; ++i) cout << dwalkAddr[i] << " ";
        cout << endl;
        cout << "  dWritebacks: ";
        for (int i = 0; i < 3; ++i) cout << dWritebacks[i] << " ";
        cout << endl;
        cout << "  sqIdx: " << sqIdx << endl;

        cout << "Instruction access:" << endl;
        cout << "  pc: " << pc << endl;
        cout << "  isBranching: " << isBranching << endl;
        cout << "  isMisPredict: " << isMisPredict << endl;
        cout << "  fetchDepth: " << fetchDepth << endl;
        cout << "  iwalkDepth: ";
        for (int i = 0; i < 3; ++i) cout << iwalkDepth[i] << " ";
        cout << endl;
        cout << "  iwalkAddr: ";
        for (int i = 0; i < 3; ++i) cout << iwalkAddr[i] << " ";
        cout << endl;
        cout << "  iWritebacks: ";
        for (int i = 0; i < 2; ++i) cout << iWritebacks[i] << " ";
        cout << endl;

        cout << "Timing:" << endl;
        cout << "  inTick: " << inTick << endl;
        cout << "  completeTick: " << completeTick << endl;
        cout << "  outTick: " << outTick << endl;
        cout << "  decodeTick: " << decodeTick << endl;
        cout << "  renameTick: " << renameTick << endl;
        cout << "  dispatchTick: " << dispatchTick << endl;
        cout << "  issueTick: " << issueTick << endl;
        cout << "  storeTick: " << storeTick << endl;
        cout << "  sqOutTick: " << sqOutTick << endl;
    }
};

#endif
