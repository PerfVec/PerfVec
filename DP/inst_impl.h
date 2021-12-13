#ifndef __INST_IMPL_Q_H__
#define __INST_IMPL_Q_H__

#define SEQ_LEN 1024
#define MAX_DIS 1000
#define MAX_LDIS 100000000

enum trainDataIdx {
  TGT_FETCH = 0,
  TGT_COMMIT,
  TGT_ST_COMMIT,
  TGT_L1_I,
  TGT_L1_LD,
  TGT_L1_ST,
  TGT_L2_I,
  TGT_L2_LD,
  TGT_L2_ST,
  TGT_MEM_I,
  TGT_MEM_LD,
  TGT_MEM_ST,
  IN_BEGIN, // 12
  IN_OP = IN_BEGIN,
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
  IN_MEMBAR,
  IN_NON_SPEC,
  IN_BRANCHING,
  IN_MIS_PRED,
  IN_FETCH_LDIS,
  IN_FETCH_SDIS,
  IN_DATA,
  IN_DATA_LLDDIS,
  IN_DATA_LSTDIS,
  IN_DATA_SDIS,
  IN_DATA_SDIS_LINE,
  IN_REG_SRC_BEGIN, // 34
  IN_REG_SRC_END = IN_REG_SRC_BEGIN + 2*SRCREGNUM - 1,
  IN_REG_DST_BEGIN, // 50
  IN_REG_DST_END = IN_REG_DST_BEGIN + 2*DSTREGNUM - 1,
  TRAIN_INST_LEN // 62
};

inline Addr getLine(Addr in) { return in & ~0x3f; }
inline int getReg(int C, int I) { return C * MAXREGIDX + I + 1; }

Tick Inst::read(ifstream &ROBtrace, ifstream &SQtrace) {
  ROBtrace >> dec >> sqIdx;
  if (ROBtrace.eof()) {
    int tmp;
    SQtrace >> tmp;
    assert(SQtrace.eof());
    return FILE_END;
  }
  ifstream *trace = &ROBtrace;
  int sqIdx2;
  Tick inTick2;
  int completeTick2, outTick2;
  if (sqIdx == -99) {
    Tick startTick;
    ROBtrace >> startTick;
    startTick /= TICK_STEP;
    assert(startTick > FILE_END);
    return startTick;
  } else {
    ROBtrace >> inTick >> completeTick >> outTick;
    assert(outTick >= completeTick);
    inTick /= TICK_STEP;
    completeTick /= TICK_STEP;
    outTick /= TICK_STEP;
    if (completeTick < minCompleteLat)
      minCompleteLat = completeTick;
    if (sqIdx != -1) {
      SQtrace >> dec >> sqIdx2 >> inTick2 >> completeTick2 >> outTick2 >>
          storeTick >> sqOutTick;
      if (SQtrace.eof())
        return FILE_END;
      trace = &SQtrace;
      assert(sqIdx2 == sqIdx && inTick2 == inTick * TICK_STEP);
      assert(sqOutTick >= storeTick);
      storeTick /= TICK_STEP;
      sqOutTick /= TICK_STEP;
      if (storeTick < minStoreLat)
        minStoreLat = storeTick;
    } else {
      storeTick = 0;
      sqOutTick = 0;
    }
  }

  // Read instruction type and etc.
  *trace >> op >> isMicroOp >> isCondCtrl >> isUncondCtrl >> isDirectCtrl >>
      isSquashAfter >> isSerializeAfter >> isSerializeBefore;
  *trace >> isAtomic >> isStoreConditional >> isMemBar >> isQuiesce >>
      isNonSpeculative;
  assert((isMicroOp == 0 || isMicroOp == 1) &&
         (isCondCtrl == 0 || isCondCtrl == 1) &&
         (isUncondCtrl == 0 || isUncondCtrl == 1) &&
         (isDirectCtrl == 0 || isDirectCtrl == 1) &&
         (isSquashAfter == 0 || isSquashAfter == 1) &&
         (isSerializeAfter == 0 || isSerializeAfter == 1) &&
         (isSerializeBefore == 0 || isSerializeBefore == 1) &&
         isAtomic == 0 &&
         (isStoreConditional == 0 || isStoreConditional == 1) &&
         (isMemBar == 0 || isMemBar == 1) &&
         isQuiesce == 0 &&
         (isNonSpeculative == 0 || isNonSpeculative == 1));
  assert(!inSQ() || (completeTick2 == completeTick * TICK_STEP &&
                     outTick2 == outTick * TICK_STEP));

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
  fprintf(stderr, "OP: %d %d %d %d %d %d %d : %d %d %d %d %d %d\n", i->op,
          i->isUncondCtrl, i->isCondCtrl, i->isDirectCtrl, i->isSquashAfter,
          i->isSerializeBefore, i->isSerializeAfter, i->isAtomic,
          i->isStoreConditional, i->isQuiesce, i->isNonSpeculative,
          i->isMemBar, i->isMisPredict);
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

void Inst::dump(Tick startTick, int *out) {
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
  } else
    out[TGT_ST_COMMIT] = 0;

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

  // Dump operations.
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
  out[IN_SC] = isStoreConditional;
  out[IN_MEMBAR] = isMemBar;
  out[IN_NON_SPEC] = isNonSpeculative;

  out[IN_BRANCHING] = isBranching;
  out[IN_MIS_PRED] = isMisPredict;
  // Instruction cache distance in the long history.
#ifdef UNIQUE_RD
  out[IN_FETCH_LDIS] = getUniqueRD(getLine(pc), pcArr, MAX_LDIS);
  pcArr.push_back(getLine(pc));
#else
  out[IN_FETCH_LDIS] = getReuseDistance(getLine(pc), instIdx, pcLMap, MAX_LDIS);
  pcLMap[getLine(pc)] = instIdx;
#endif
  // Instruction cache line distance.
  out[IN_FETCH_SDIS] =
      getReuseDistance(getLine(pc), curInstNum, pcMap, MAX_DIS);
  pcMap[getLine(pc)] = curInstNum;

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
    out[IN_DATA_SDIS] = getReuseDistance(addr, curInstNum, dataMap, MAX_DIS);
    // FIXME: number of memory accesses instead?
    dataMap[addr] = curInstNum;
    // Data address cache line distance.
    out[IN_DATA_SDIS_LINE] =
        getReuseDistance(getLine(addr), curInstNum, dataLineMap, MAX_DIS);
    dataLineMap[getLine(addr)] = curInstNum;
  } else {
    out[IN_DATA_LLDDIS] = 0;
    out[IN_DATA_LSTDIS] = 0;
    out[IN_DATA_SDIS] = 0;
    out[IN_DATA_SDIS_LINE] = 0;
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

#endif
