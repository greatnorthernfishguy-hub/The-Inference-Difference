import sys

NL = chr(10)
p = '/home/josh/The-Inference-Difference/inference_difference/router.py'
ls = open(p).readlines()

# 1. Changelog
for i, l in enumerate(ls):
    if '# ---- Changelog ----' in l:
        if '#173' not in ls[i+1]:
            ls.insert(i+1, '# [2026-04-19] CC Sonnet 4.6 -- #173: cascade avoidance (a-d)' + NL)
        break

# 2. Success stats fields after _history_max
for i, l in enumerate(ls):
    if 'self._history_max = 500' in l:
        if '_model_success_stats' not in ls[i+1]:
            ls.insert(i+1, '        self._model_success_stats = {}  # model_id -> list[bool]' + NL)
            ls.insert(i+2, '        self._SUCCESS_WINDOW = 50' + NL)
            ls.insert(i+3, '        self._SUCCESS_FLOOR = 0.20' + NL)
            ls.insert(i+4, '        self._SUCCESS_MIN_SAMPLES = 10' + NL)
        break

# 3. Record success stats in report_outcome
for i, l in enumerate(ls):
    if 'self._decision_history[-self._history_max:]' in l:
        if '_model_success_stats' not in ls[i+1]:
            ls.insert(i+1, '        st = self._model_success_stats.setdefault(decision.model_id, [])' + NL)
            ls.insert(i+2, '        st.append(success)' + NL)
            ls.insert(i+3, '        if len(st) > self._SUCCESS_WINDOW:' + NL)
            ls.insert(i+4, '            self._model_success_stats[decision.model_id] = st[-self._SUCCESS_WINDOW:]' + NL)
        break

# 4. Success-rate floor in _filter_candidates
for i, l in enumerate(ls):
    if 'return candidates, quality_floor_bypassed' in l:
        if '_SUCCESS_FLOOR' not in ls[i-1]:
            ls.insert(i, '        if candidates:' + NL)
            ls.insert(i+1, '            sf, pr = [], 0' + NL)
            ls.insert(i+2, '            for m in candidates:' + NL)
            ls.insert(i+3, '                ms = self._model_success_stats.get(m.model_id, [])' + NL)
            ls.insert(i+4, '                if len(ms) >= self._SUCCESS_MIN_SAMPLES and sum(ms)/len(ms) < self._SUCCESS_FLOOR:' + NL)
            ls.insert(i+5, '                    pr += 1; continue' + NL)
            ls.insert(i+6, '                sf.append(m)' + NL)
            ls.insert(i+7, '            if sf:' + NL)
            ls.insert(i+8, '                if pr: logger.info("Success-rate floor pruned %d model(s)", pr)' + NL)
            ls.insert(i+9, '                candidates = sf' + NL)
        break

# 5. Replace _score_cost with success_rate-aware version
start, end = None, None
for i, l in enumerate(ls):
    if 'def _score_cost(self, model: ModelEntry)' in l:
        start = i
    elif start is not None and i > start and l.startswith('    def '):
        end = i
        break
if start is not None and end is not None and 'success_rate' not in ls[start]:
    new_fn = [
        '    def _score_cost(self, model: ModelEntry, success_rate: float = 1.0) -> float:' + NL,
        '        ' + '"""' + 'Expected true cost = raw_price / success_rate (#173a).' + NL,
        '' + NL,
        '        Free models use 0.001/1k synthetic so low-success free models score correctly.' + NL,
        '        ' + '"""' + NL,
        '        success_rate = max(success_rate, 0.01)' + NL,
        '        raw_cost = model.cost_per_1k_tokens if model.cost_per_1k_tokens > 0 else 0.001' + NL,
        '        effective_cost = raw_cost / success_rate' + NL,
        '        if model.cost_per_1k_tokens <= 0 and success_rate >= 0.99:' + NL,
        '            return 1.0  # local model with full success -- still free' + NL,
        '' + NL,
        '        budget = self.config.cost_budget_per_request' + NL,
        '        if budget <= 0:' + NL,
        '            return self.config.neutral_score' + NL,
        '' + NL,
        '        cost_fraction = effective_cost / budget' + NL,
        '        return max(0.0, 1.0 - cost_fraction)' + NL,
        '' + NL,
    ]
    ls[start:end] = new_fn

# 6. Pass success_rate to _score_cost in _score_model
for i, l in enumerate(ls):
    if 'cost_score = self._score_cost(model)' in l:
        ls[i] = '        _ms = self._model_success_stats.get(model.model_id, [])' + NL
        ls.insert(i+1, '        _sr = (sum(_ms)/len(_ms)) if len(_ms) >= self._SUCCESS_MIN_SAMPLES else 1.0' + NL)
        ls.insert(i+2, '        cost_score = self._score_cost(model, success_rate=_sr)' + NL)
        break

open(p, 'w').writelines(ls)
print('ROUTER_PATCHED')
