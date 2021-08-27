#ifndef __INST_IMPL_Q_H__
#define __INST_IMPL_Q_H__

#define MAX_DIS 1000

Addr getLine(Addr in) { return in & ~0x3f; }
int getReg(int C, int I) { return C * MAXREGIDX + I + 1; }

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
      assert(sqOutTick + inTick >= curSqOutTick);
      curSqOutTick = inTick + sqOutTick;
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
  *trace >> dec >> isMisPredict >> fetchDepth;
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

void Inst::dump(Tick startTick, int *out) {
  Tick fetchLat = inTick - startTick;
  out[0] = fetchLat;
  out[1] = completeTick + fetchLat;
  out[2] = outTick + fetchLat;
  if (storeTick > 0)
    out[3] = storeTick + fetchLat;
  else
    out[3] = 0;
  if (curSqOutTick > 0)
    out[4] = curSqOutTick - startTick;
  else
    out[4] = 0;

  // Dump operations.
  out[5] = op + 1;
  out[6] = inSQ();
  out[7] = isMicroOp;
  out[8] = isMisPredict;
  out[9] = isCondCtrl;
  out[10] = isUncondCtrl;
  out[11] = isDirectCtrl;
  out[12] = isSquashAfter;
  out[13] = isSerializeAfter;
  out[14] = isSerializeBefore;
  out[15] = isStoreConditional;
  out[16] = isMemBar;
  out[17] = isNonSpeculative;

  // Instruction cache depth.
  out[18] = fetchDepth;
  // Instruction cache line distance.
  auto mapIter = pcMap.find(getLine(pc));
  if (mapIter == pcMap.end())
    out[19] = MAX_DIS;
  else {
    out[19] = curInstNum - mapIter->second;
    if (out[19] > MAX_DIS)
      out[19] = MAX_DIS;
    assert(out[19] > 0);
  }
  pcMap[getLine(pc)] = curInstNum;

  out[20] = isAddr;
  // Data cache depth.
  out[21] = depth;
  if (isAddr) {
    // Data address distance.
    mapIter = dataMap.find(addr);
    if (mapIter == dataMap.end())
      out[22] = MAX_DIS;
    else {
      out[22] = curInstNum - mapIter->second;
      if (out[22] > MAX_DIS)
        out[22] = MAX_DIS;
      assert(out[22] > 0);
    }
    dataMap[addr] = curInstNum;
    // Data address cache line distance.
    mapIter = dataLineMap.find(getLine(addr));
    if (mapIter == dataLineMap.end())
      out[23] = MAX_DIS;
    else {
      out[23] = curInstNum - mapIter->second;
      if (out[23] > MAX_DIS)
        out[23] = MAX_DIS;
      assert(out[23] > 0);
    }
    dataLineMap[getLine(addr)] = curInstNum;
  } else {
    out[22] = 0;
    out[23] = 0;
  }

  // Registers.
  for (int i = 0; i < srcNum; i++) {
    out[24 + 2*i] = srcClass[i] + 1;
    // FIXME: replace with distance?
    out[24 + 2*i+1] = srcIndex[i] + 1;
  }
  for (int i = srcNum; i < SRCREGNUM; i++) {
    out[24 + 2*i] = 0;
    // FIXME: replace with distance? Max in this case.
    out[24 + 2*i+1] = 0;
  }
  for (int i = 0; i < destNum; i++) {
    out[24 + SRCREGNUM*2 + 2*i] = destClass[i] + 1;
    out[24 + SRCREGNUM*2 + 2*i+1] = destIndex[i] + 1;
  }
  for (int i = destNum; i < DSTREGNUM; i++) {
    out[24 + SRCREGNUM*2 + 2*i] = 0;
    out[24 + SRCREGNUM*2 + 2*i+1] = 0;
  }
}

#endif
