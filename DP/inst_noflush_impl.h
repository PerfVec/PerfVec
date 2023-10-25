#ifndef __INST_IMPL_Q_H__
#define __INST_IMPL_Q_H__

#include <cmath>
#include "reuse-dist.h"

//#define UNIQUE_RD
#define TREE_RD

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
#elif defined(TREE_RD)
ReuseDistance *PCRD;
ReuseDistance *LdRD;
ReuseDistance *StRD;
#else
unordered_map<Addr, Tick> pcLMap;
unordered_map<Addr, Tick> dataLineLdMap;
unordered_map<Addr, Tick> dataLineStMap;
#endif

//#define TICK_STEP 500
#define TICK_STEP 100
#define MAX_DIS 1000
#define MAX_LDIS 100000000

enum targetIdx {
  TGT_FETCH = 0,
  TGT_COMMIT,
  TGT_ST_COMMIT,
  TGT_DECODE,
  TGT_RENAME,
  TGT_DISPATCH,
  //TGT_ISSUE,
  TGT_L1_I,
  TGT_L1_LD,
  TGT_L1_ST,
  TGT_L2_I,
  TGT_L2_LD,
  TGT_L2_ST,
  TGT_MEM_I,
  TGT_MEM_LD,
  TGT_MEM_ST,
  TGT_MIS_PRED,
  TGT_LEN // 16
};

enum featureIdx {
  IN_BEGIN = 0,
  IN_FAULT = IN_BEGIN,
  IN_OP,
  IN_ST,
  IN_LD,
  IN_MICRO,
  IN_COND_CTRL,
  IN_UNCOND_CTRL,
  IN_DIRECT_CTRL,
  IN_SQUASH_AF,
  IN_SERIAL_AF,
  IN_SERIAL_BE,
  IN_SC,
  IN_RDBAR,
  IN_WRBAR,
  IN_NON_SPEC,
  IN_BRANCHING,
  IN_GBE_PRE,
  IN_GBE_NOW,
  IN_FETCH_LDIS,
  IN_DATA,
  IN_DATA_LLDDIS,
  IN_DATA_LSTDIS,
  IN_DATA_SDIS,
  IN_REG_SRC_BEGIN, // 23
  IN_REG_SRC_END = IN_REG_SRC_BEGIN + 2*SRCREGNUM - 1,
  IN_REG_DST_BEGIN, // 39
  IN_REG_DST_END = IN_REG_DST_BEGIN + 2*DSTREGNUM - 1,
  IN_LEN // 51
};

inline Addr getLine(Addr in) { return in & ~0x3f; }
inline int getReg(int C, int I) { return C * MAXREGIDX + I + 1; }

Tick Inst::read(ifstream &ROBtrace, ifstream &SQtrace, bool isSingleTrace) {
  ROBtrace >> dec >> isFault >> sqIdx;
  if (ROBtrace.eof()) {
    if (!isSingleTrace) {
      int tmp;
      SQtrace >> tmp;
      assert(SQtrace.eof());
    }
    return FILE_END;
  }
  ifstream *trace = &ROBtrace;
  Tick completeTick2, outTick2;
  if (sqIdx == -99) {
    Tick startTick;
    ROBtrace >> startTick;
    startTick /= TICK_STEP;
    assert(startTick > FILE_END);
    return startTick;
  } else {
    ROBtrace >> inTick >> completeTick >> outTick;
    ROBtrace >> decodeTick >> renameTick >> dispatchTick >> issueTick;
    assert(outTick >= completeTick);
    if (isSingleTrace) {
      ROBtrace >> storeTick >> sqOutTick;
    } else if (sqIdx != -1 && !isFault && ROBtrace.peek() != '\n') {
      ROBtrace >> storeTick >> sqOutTick;
    } else if (sqIdx != -1 && !isFault) {
      int isFault2;
      long sqIdx2;
      Tick inTick2;
      Tick decodeTick2, renameTick2, dispatchTick2, issueTick2;
      SQtrace >> isFault2 >> sqIdx2 >> inTick2 >> completeTick2 >> outTick2 >>
          decodeTick2 >> renameTick2 >> dispatchTick2 >> issueTick2 >>
          storeTick >> sqOutTick;
      if (SQtrace.eof())
        return FILE_END;
      assert(isFault2 == isFault && (sqIdx2 == sqIdx || sqIdx == 0) &&
             inTick2 == inTick && decodeTick2 == decodeTick &&
             renameTick2 == renameTick && dispatchTick2 == dispatchTick &&
             issueTick2 == issueTick);
      assert(sqOutTick >= storeTick);
      trace = &SQtrace;
    } else {
      storeTick = 0;
      sqOutTick = 0;
    }
    inTick /= TICK_STEP;
    completeTick /= TICK_STEP;
    outTick /= TICK_STEP;
    if (completeTick < minCompleteLat)
      minCompleteLat = completeTick;
    storeTick /= TICK_STEP;
    sqOutTick /= TICK_STEP;
    if (storeTick != 0 && storeTick < minStoreLat)
      minStoreLat = storeTick;
    decodeTick /= TICK_STEP;
    renameTick /= TICK_STEP;
    dispatchTick /= TICK_STEP;
    issueTick /= TICK_STEP;
  }

  // Read instruction type and etc.
  *trace >> op >> isMicroOp >> isCondCtrl >> isUncondCtrl >> isDirectCtrl >>
      isSquashAfter >> isSerializeAfter >> isSerializeBefore;
  *trace >> isAtomic >> isStoreConditional >> isRdBar >> isWrBar >> isQuiesce >>
      isNonSpeculative;
  assert((isMicroOp == 0 || isMicroOp == 1) &&
         (isCondCtrl == 0 || isCondCtrl == 1) &&
         (isUncondCtrl == 0 || isUncondCtrl == 1) &&
         (isDirectCtrl == 0 || isDirectCtrl == 1) &&
         (isSquashAfter == 0 || isSquashAfter == 1) &&
         (isSerializeAfter == 0 || isSerializeAfter == 1) &&
         (isSerializeBefore == 0 || isSerializeBefore == 1) &&
         (isAtomic == 0 || isAtomic == 1) &&
         (isStoreConditional == 0 || isStoreConditional == 1) &&
         (isRdBar == 0 || isRdBar == 1) &&
         (isWrBar == 0 || isWrBar == 1) &&
         isQuiesce == 0 &&
         (isNonSpeculative == 0 || isNonSpeculative == 1));
  assert(!inSQ() || isSingleTrace || isFault ||
         (completeTick2 / TICK_STEP == completeTick &&
          outTick2 / TICK_STEP == outTick));

  // Read data memory access info.
  *trace >> isAddr;
  *trace >> dec >> addr;
  *trace >> dec >> size >> depth;
  if (isAddr) {
    addrEnd = addr + size - 1;
  } else
    addrEnd = 0;
  for (int i = 0; i < 3; i++)
    *trace >> dwalkDepth[i];
  for (int i = 0; i < 3; i++) {
    *trace >> dec >> dwalkAddr[i];
    assert(dwalkAddr[i] == 0 && dwalkDepth[i] == -1);
  }
  for (int i = 0; i < 3; i++)
    *trace >> dec >> dWritebacks[i];

  // Read instruction memory access info.
  *trace >> dec >> pc;
  *trace >> dec >> isBranching >> isMisPredict >> fetchDepth;
  for (int i = 0; i < 3; i++)
    *trace >> iwalkDepth[i];
  for (int i = 0; i < 3; i++) {
    *trace >> dec >> iwalkAddr[i];
    assert(iwalkAddr[i] == 0 && iwalkDepth[i] == -1);
  }
  for (int i = 0; i < 2; i++)
    *trace >> dec >> iWritebacks[i];

  // Read source and destination registers.
  *trace >> srcNum >> destNum;
  assert(srcNum <= SRCREGNUM && destNum <= DSTREGNUM);
  for (int i = 0; i < srcNum; i++) {
    *trace >> srcClass[i] >> srcIndex[i];
    assert(srcClass[i] <= MAXREGCLASS);
    assert(srcClass[i] == MAXREGCLASS || srcIndex[i] < MAXREGIDX);
  }
  for (int i = 0; i < destNum; i++) {
    *trace >> destClass[i] >> destIndex[i];
    assert(destClass[i] <= MAXREGCLASS);
    assert(destClass[i] == MAXREGCLASS || destIndex[i] < MAXREGIDX);
  }

  assert(!ROBtrace.eof() && !SQtrace.eof());
  return READ_INST;
}

void printOP(Inst *i) {
  fprintf(stderr, "OP: %d %d %d %d %d %d %d %d : %d %d %d %d %d %d %d\n",
          i->isFault, i->op, i->isUncondCtrl, i->isCondCtrl, i->isDirectCtrl,
          i->isSquashAfter, i->isSerializeBefore, i->isSerializeAfter,
          i->isAtomic, i->isStoreConditional, i->isQuiesce, i->isNonSpeculative,
          i->isRdBar, i->isWrBar, i->isMisPredict);
}

template <class T>
int getReuseDistance(Addr addr, T idx, unordered_map<Addr, T> &map, int max) {
  int res;
  auto mapLIter = map.find(addr);
  if (mapLIter == map.end())
    res = max;
  else {
    assert(idx > mapLIter->second);
    Tick dis = idx - mapLIter->second;
    if (dis > max)
      res = max;
    else
      res = dis;
  }
  return res;
}

int getUniqueRD(Addr addr, vector<Addr> &arr, int max) {
  unordered_map<Addr, bool> map;
  for (auto i = arr.end(); i != arr.begin(); i--) {
    if (addr == *i)
      return map.size() + 1;
    else {
      map[*i] = true;
      if (map.size() >= max)
        return max;
    }
  }
  return max;
}

class BranchEntropy {
public:
  double globalPre;
  double globalNow;

  BranchEntropy() {
    globalPre = 0.0;
    globalNow = 0.0;
  }
};

unordered_map<Addr, vector<bool> *> BranchHistories;

#define BRANCH_HISTORY_LENGTH 8

BranchEntropy getBranchEntropy(Addr pc, bool taken) {
  BranchEntropy BE;
  auto hisIter = BranchHistories.find(pc);
  if (hisIter == BranchHistories.end()) {
    BranchHistories[pc] = new vector<bool>();
    hisIter = BranchHistories.find(pc);
  }
  hisIter->second->push_back(taken);
  int hisSize = hisIter->second->size();
  // Compute the global entropy.
  for (int i = 0; i < BRANCH_HISTORY_LENGTH+1; i++) {
    double hisTaken;
    if (hisSize - 1 - i >= 0)
      hisTaken = hisIter->second->at(hisSize - 1 - i);
    else
      hisTaken = 0.5;
    if (i < BRANCH_HISTORY_LENGTH)
      BE.globalNow += hisTaken;
    if (i > 0)
      BE.globalPre += hisTaken;
  }
  double p = BE.globalNow / BRANCH_HISTORY_LENGTH;
  assert(p >= 0.0 && p <= 1.0);
  if (p == 0.0 || p == 1.0)
    BE.globalNow = 0.0;
  else
    BE.globalNow = -p * log(p) - (1-p) * log(1-p);
  p = BE.globalPre / BRANCH_HISTORY_LENGTH;
  assert(p >= 0.0 && p <= 1.0);
  if (p == 0.0 || p == 1.0)
    BE.globalPre = 0.0;
  else
    BE.globalPre = -p * log(p) - (1-p) * log(1-p);
  // FIXME: history aware entropy.
  return BE;
}

void Inst::dumpTargets(Tick startTick, double *out) {
  dumpTargets(startTick, out, memLdIdx, memStIdx, lastFetchTick, lastCommitTick,
              lastSqOutTick, lastDecodeTick, lastRenameTick, lastDispatchTick);
}

void Inst::dumpTargets(Tick startTick, double *out, Tick &memLdIdx,
                       Tick &memStIdx, Tick &lastFetchTick,
                       Tick &lastCommitTick, Tick &lastSqOutTick,
                       Tick &lastDecodeTick, Tick &lastRenameTick,
                       Tick &lastDispatchTick) {
  // Calculate target latencies.
  assert(inTick >= startTick);
  Tick fetchLat = inTick - startTick;
  assert(fetchLat >= lastFetchTick);
  out[TGT_FETCH] = fetchLat - lastFetchTick;
  lastFetchTick = fetchLat;
  assert(outTick + fetchLat >= lastCommitTick);
  out[TGT_COMMIT] = outTick + fetchLat - lastCommitTick;
  lastCommitTick = outTick + fetchLat;
  if (sqOutTick > 0) {
    assert(sqOutTick + fetchLat >= lastSqOutTick);
    out[TGT_ST_COMMIT] = sqOutTick + fetchLat - lastSqOutTick;
    lastSqOutTick = sqOutTick + fetchLat;
  } else {
    //out[TGT_ST_COMMIT] = 0;
    //out[TGT_ST_COMMIT] = outTick + fetchLat - lastSqOutTick;
    //lastSqOutTick = outTick + fetchLat;
    if (lastCommitTick > lastSqOutTick) {
      out[TGT_ST_COMMIT] = lastCommitTick - lastSqOutTick;
      lastSqOutTick = lastCommitTick;
    } else
      out[TGT_ST_COMMIT] = 0;
  }
  assert(decodeTick + fetchLat >= lastDecodeTick);
  out[TGT_DECODE] = decodeTick + fetchLat - lastDecodeTick;
  lastDecodeTick = decodeTick + fetchLat;
  assert(renameTick + fetchLat >= lastRenameTick);
  out[TGT_RENAME] = renameTick + fetchLat - lastRenameTick;
  lastRenameTick = renameTick + fetchLat;
  assert(dispatchTick + fetchLat >= lastDispatchTick);
  out[TGT_DISPATCH] = dispatchTick + fetchLat - lastDispatchTick;
  lastDispatchTick = dispatchTick + fetchLat;
  //out[TGT_ISSUE] = issueTick + fetchLat - lastIssueTick;
  //lastIssueTick = issueTick + fetchLat;

  // Calculate target cache/memory accesses.
  assert(fetchDepth >= -1 && fetchDepth <= 2);
  if (fetchDepth == 2)
    out[TGT_MEM_I] = 1;
  else
    out[TGT_MEM_I] = 0;
  if (fetchDepth >= 1)
    out[TGT_L2_I] = 1;
  else
    out[TGT_L2_I] = 0;
  if (fetchDepth >= 0)
    out[TGT_L1_I] = 1;
  else
    out[TGT_L1_I] = 0;

  assert(depth >= -1 && depth <= 2);
  out[TGT_MEM_LD] = 0;
  out[TGT_MEM_ST] = 0;
  if (depth == 2) {
    if (isLoad())
      out[TGT_MEM_LD] = 1;
    else {
      assert(isStore());
      out[TGT_MEM_ST] = 1;
    }
  }
  out[TGT_L2_LD] = 0;
  out[TGT_L2_ST] = 0;
  if (depth >= 1) {
    if (isLoad())
      out[TGT_L2_LD] = 1;
    else {
      assert(isStore());
      out[TGT_L2_ST] = 1;
    }
  }
  out[TGT_L1_LD] = 0;
  out[TGT_L1_ST] = 0;
  if (depth >= 0) {
    if (isLoad())
      out[TGT_L1_LD] = 1;
    else {
      assert(isStore());
      out[TGT_L1_ST] = 1;
    }
  }
  out[TGT_MIS_PRED] = isMisPredict;
}

void Inst::dumpFeatures(Tick startTick, double *out) {
  // Dump operations.
  out[IN_FAULT] = isFault;
  out[IN_OP]  = op + 1;
  out[IN_ST]  = inSQ();
  out[IN_LD]  = isLoad();
  out[IN_MICRO]  = isMicroOp;
  out[IN_COND_CTRL]  = isCondCtrl;
  out[IN_UNCOND_CTRL]  = isUncondCtrl;
  out[IN_DIRECT_CTRL] = isDirectCtrl;
  out[IN_SQUASH_AF] = isSquashAfter;
  out[IN_SERIAL_AF] = isSerializeAfter;
  out[IN_SERIAL_BE] = isSerializeBefore;
  out[IN_SC] = isStoreConditional + isAtomic * 2;
  out[IN_RDBAR] = isRdBar;
  out[IN_WRBAR] = isWrBar;
  out[IN_NON_SPEC] = isNonSpeculative;

  out[IN_BRANCHING] = isBranching;
  //out[IN_MIS_PRED] = isMisPredict;
  if (isCondCtrl) {
    BranchEntropy BE = getBranchEntropy(pc, isBranching);
    out[IN_GBE_PRE] = BE.globalPre;
    out[IN_GBE_NOW] = BE.globalNow;
  } else {
    out[IN_GBE_PRE] = 0.0;
    out[IN_GBE_NOW] = 0.0;
  }
  // Instruction cache distance in the long history.
#ifdef UNIQUE_RD
  out[IN_FETCH_LDIS] = getUniqueRD(getLine(pc), pcArr, MAX_LDIS);
  pcArr.push_back(getLine(pc));
#elif defined(TREE_RD)
  out[IN_FETCH_LDIS] = PCRD->process_address(getLine(pc), true);
#else
  out[IN_FETCH_LDIS] = getReuseDistance(getLine(pc), instIdx, pcLMap, MAX_LDIS);
  pcLMap[getLine(pc)] = instIdx;
#endif

  out[IN_DATA] = isAddr;
  if (isAddr) {
    // Data cache distance in the long history.
#ifdef UNIQUE_RD
    out[IN_DATA_LLDDIS] = getUniqueRD(getLine(addr), dataLdArr, MAX_LDIS);
    out[IN_DATA_LSTDIS] = getUniqueRD(getLine(addr), dataStArr, MAX_LDIS);
    if (isLoad())
      dataLdArr.push_back(getLine(addr));
    else {
      assert(isStore());
      dataStArr.push_back(getLine(addr));
    }
#elif defined(TREE_RD)
    bool LdUpdate = isLoad();
    out[IN_DATA_LLDDIS] = LdRD->process_address(getLine(addr), LdUpdate);
    out[IN_DATA_LSTDIS] = StRD->process_address(getLine(addr), !LdUpdate);
#else
    out[IN_DATA_LLDDIS] =
        getReuseDistance(getLine(addr), memLdIdx, dataLineLdMap, MAX_LDIS);
    out[IN_DATA_LSTDIS] =
        getReuseDistance(getLine(addr), memStIdx, dataLineStMap, MAX_LDIS);
    if (isLoad())
      dataLineLdMap[getLine(addr)] = memLdIdx;
    else {
      assert(isStore());
      dataLineStMap[getLine(addr)] = memStIdx;
    }
#endif
    // Data address distance.
    // FIXME: separate load and store?
    out[IN_DATA_SDIS] = getReuseDistance(addr, instIdx, dataMap, MAX_DIS);
    // FIXME: number of memory accesses instead?
    dataMap[addr] = instIdx;
  } else {
    out[IN_DATA_LLDDIS] = 0;
    out[IN_DATA_LSTDIS] = 0;
    out[IN_DATA_SDIS] = 0;
  }

  // Registers.
  for (int i = 0; i < srcNum; i++) {
    out[IN_REG_SRC_BEGIN + 2*i] = srcClass[i] + 1;
    // FIXME: replace with distance?
    out[IN_REG_SRC_BEGIN + 2*i+1] = srcIndex[i] + 1;
  }
  for (int i = srcNum; i < SRCREGNUM; i++) {
    out[IN_REG_SRC_BEGIN + 2*i] = 0;
    // FIXME: replace with distance? Max in this case.
    out[IN_REG_SRC_BEGIN + 2*i+1] = 0;
  }
  for (int i = 0; i < destNum; i++) {
    out[IN_REG_DST_BEGIN + 2*i] = destClass[i] + 1;
    out[IN_REG_DST_BEGIN + 2*i+1] = destIndex[i] + 1;
  }
  for (int i = destNum; i < DSTREGNUM; i++) {
    out[IN_REG_DST_BEGIN + 2*i] = 0;
    out[IN_REG_DST_BEGIN + 2*i+1] = 0;
  }
}

void Inst::dump(Tick startTick, double *out) {
  dumpTargets(startTick, out);
  dumpFeatures(startTick, out + TGT_LEN);
}

#endif
