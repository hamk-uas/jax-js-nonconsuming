//#region rolldown:runtime
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __commonJS = (cb, mod$1) => function() {
	return mod$1 || (0, cb[__getOwnPropNames(cb)[0]])((mod$1 = { exports: {} }).exports, mod$1), mod$1.exports;
};
var __export = (target, all$1) => {
	for (var name in all$1) __defProp(target, name, {
		get: all$1[name],
		enumerable: true
	});
};
var __copyProps = (to, from, except, desc) => {
	if (from && typeof from === "object" || typeof from === "function") for (var keys = __getOwnPropNames(from), i = 0, n = keys.length, key$1; i < n; i++) {
		key$1 = keys[i];
		if (!__hasOwnProp.call(to, key$1) && key$1 !== except) __defProp(to, key$1, {
			get: ((k) => from[k]).bind(null, key$1),
			enumerable: !(desc = __getOwnPropDesc(from, key$1)) || desc.enumerable
		});
	}
	return to;
};
var __toESM = (mod$1, isNodeMode, target) => (target = mod$1 != null ? __create(__getProtoOf(mod$1)) : {}, __copyProps(isNodeMode || !mod$1 || !mod$1.__esModule ? __defProp(target, "default", {
	value: mod$1,
	enumerable: true
}) : target, mod$1));

//#endregion
const require_backend = require('./backend-DR8VahhR.cjs');

//#region src/frontend/check-leaks.ts
const _jitCacheDisposers = [];
/** Register a callback that clears a JIT-related cache. Called by jit.ts. */
function _registerJitCacheDisposer(fn) {
	_jitCacheDisposers.push(fn);
}
/**
* Per-function jit() cache disposers. Each jit() call registers a callback
* that disposes its ClosedJaxpr consts and clears its cache. This allows
* `_disposeAllJitCaches()` to clean up module-level jit functions whose consts
* were created during a test session.
*/
const _jitFunctionDisposers = /* @__PURE__ */ new Set();
function _disposeAllJitCaches() {
	for (const fn of _jitCacheDisposers) fn();
	for (const fn of _jitFunctionDisposers) fn();
}
/** True only between start() and stop(). Checked in Array ctor/dispose. */
let _leakTrackingEnabled = false;
/**
* Live tracked Arrays → Error captured at construction time.
* `new Error()` records V8 frame pointers cheaply; the `.stack` string
* is only formatted on first access (cold path, inside stop()).
*/
const _leakTrackingMap = /* @__PURE__ */ new Map();
/**
* When `trackRefs` is enabled, records the last `.ref` call site per array.
* Only populated when `start({ trackRefs: true })` — checked in array.ts.
*/
let _trackRefsEnabled = false;
const _lastRefMap = /* @__PURE__ */ new Map();
const _startSlots = /* @__PURE__ */ new Map();
let _active = false;
/**
* Library source URL prefix, derived from this module's own URL.
* Any stack frame whose URL starts with this is library-internal.
* Computed once at load time (not on the hot path).
*/
const _libSrcPrefix = (() => {
	try {
		const url = require("url").pathToFileURL(__filename).href;
		const i = url.lastIndexOf("/src/frontend/check-leaks.");
		if (i >= 0) return url.slice(0, i + 1) + "src/";
	} catch {}
	return null;
})();
/**
* Detect which package a raw frame URL belongs to.
* Returns the package name (e.g. "@jax-js/jax", "some-lib") or null for user code.
*/
function detectPackage(rawUrl) {
	const nm = rawUrl.match(/node_modules\/((?:@[^/]+\/)?[^/]+)/);
	if (nm) return nm[1];
	if (_libSrcPrefix !== null && rawUrl.startsWith(_libSrcPrefix)) return "@jax-js/jax";
	return null;
}
/** Convert a Vite/V8 stack location to a workspace-relative path. */
function toRelativePath(raw) {
	let loc = raw.replace(/\?[^:]*(?=:\d+:\d+)/, "");
	loc = loc.replace(/^https?:\/\/[^/]+\//, "");
	if (loc.startsWith("@fs/")) {
		loc = loc.slice(4);
		if (!loc.startsWith("/")) loc = "/" + loc;
	}
	loc = loc.replace(/^file:\/\//, "");
	for (const m of [
		"/src/",
		"/test/",
		"/packages/",
		"/bench/"
	]) {
		const i = loc.indexOf(m);
		if (i >= 0) return loc.slice(i + 1);
	}
	return loc;
}
/** Boilerplate frames to skip (Array constructor internals). */
const SKIP_FUNCS = [
	"new Array",
	"new PendingExecute",
	"Array2",
	"new Array2"
];
/** Dep functions that appear in every creation stack — filtered from via chain. */
const VIA_SKIP = new Set([
	"bind",
	"bind1",
	"pureArray",
	"fullLower",
	"fullRaise"
]);
/**
* Walk the Error's stack to find the creation site.
* Accessing err.stack here triggers V8's lazy string formatting.
*/
function findCreationSite(err) {
	const stack$1 = err.stack;
	if (!stack$1) return {
		location: "(unknown)",
		via: null,
		pkg: "(unknown)"
	};
	let lastDepLoc = null;
	let lastDepPkg = null;
	/** Dep function names (innermost-first) for building via chain. */
	const depChain = [];
	let prevClean = null;
	for (const line of stack$1.split("\n")) {
		const trimmed = line.trim();
		if (!trimmed.startsWith("at ")) continue;
		let rawLoc;
		let funcName;
		const m = trimmed.match(/^at\s+(.+?)\s+\((.+)\)$/);
		if (m) {
			funcName = m[1];
			rawLoc = m[2];
		} else rawLoc = trimmed.slice(3);
		const location = toRelativePath(rawLoc);
		const pathOnly = location.replace(/:\d+:\d+$/, "");
		if (pathOnly === "<anonymous>" || pathOnly === "native" || pathOnly.startsWith("eval ")) continue;
		if (funcName && SKIP_FUNCS.some((p) => funcName.startsWith(p))) continue;
		const pkg = detectPackage(rawLoc);
		if (pkg && !/\.test\.[jt]sx?$/.test(pathOnly)) {
			lastDepLoc = location;
			lastDepPkg = pkg;
			const clean = funcName?.replace(/^(Module|Object)\./, "") ?? null;
			if (clean && clean !== prevClean && !VIA_SKIP.has(clean)) {
				depChain.push(clean);
				prevClean = clean;
			}
			continue;
		}
		const chain = depChain.reverse().slice(0, 2);
		const via = chain.length > 0 ? chain.join(" → ") : null;
		return {
			location,
			via,
			pkg: null
		};
	}
	return {
		location: lastDepLoc ?? "(unknown)",
		via: null,
		pkg: lastDepPkg ?? "(unknown)"
	};
}
function safeDesc(arr) {
	try {
		return arr.toString();
	} catch {
		return "(unknown)";
	}
}
function safeRefCountNum(arr) {
	try {
		const rc = arr.refCount;
		return typeof rc === "number" ? rc : 0;
	} catch {
		return 0;
	}
}
function safeRefCount(arr) {
	const rc = safeRefCountNum(arr);
	return rc > 0 ? ` rc=${rc}` : "";
}
const checkLeaks = {
	start(opts) {
		_leakTrackingMap.clear();
		_lastRefMap.clear();
		_trackRefsEnabled = opts?.trackRefs ?? false;
		_startSlots.clear();
		for (const d of require_backend.devices) try {
			_startSlots.set(d, require_backend.getBackend(d).slotCount());
		} catch {}
		_leakTrackingEnabled = true;
		_active = true;
	},
	snapshot() {
		const entries = [];
		for (const [arr, err] of _leakTrackingMap) {
			const site = findCreationSite(err);
			const refErr = _lastRefMap.get(arr);
			let lastRef = null;
			if (refErr) {
				const refSite = findCreationSite(refErr);
				lastRef = refSite.location;
			}
			entries.push({
				desc: safeDesc(arr),
				rc: safeRefCountNum(arr),
				location: site.location,
				pkg: site.pkg,
				lastRef
			});
		}
		return entries;
	},
	stop() {
		if (!_active) return {
			leaked: 0,
			details: [],
			summary: "No leaks detected."
		};
		_leakTrackingEnabled = false;
		_trackRefsEnabled = false;
		_active = false;
		_disposeAllJitCaches();
		let leaked = 0;
		for (const d of require_backend.devices) try {
			const baseline = _startSlots.get(d) ?? 0;
			leaked += require_backend.getBackend(d).slotCount() - baseline;
		} catch {}
		_startSlots.clear();
		if (leaked === 0) {
			_leakTrackingMap.clear();
			_lastRefMap.clear();
			return {
				leaked: 0,
				details: [],
				summary: "No leaks detected."
			};
		}
		const details = [];
		const groups = /* @__PURE__ */ new Map();
		/** Leak counts by package (null key = user code). */
		const pkgCounts = /* @__PURE__ */ new Map();
		for (const [arr, err] of _leakTrackingMap) {
			const site = findCreationSite(err);
			const desc = safeDesc(arr) + safeRefCount(arr);
			details.push(`${desc} created at ${site.location}`);
			let g = groups.get(site.location);
			if (!g) {
				g = {
					count: 0,
					descs: [],
					via: site.via,
					pkg: site.pkg,
					maxRc: 0,
					lastRefLocs: []
				};
				groups.set(site.location, g);
			}
			g.count++;
			g.maxRc = Math.max(g.maxRc, safeRefCountNum(arr));
			if (g.descs.length < 4) g.descs.push(desc);
			const refErr = _lastRefMap.get(arr);
			if (refErr && safeRefCountNum(arr) >= 2) {
				const refSite = findCreationSite(refErr);
				if (g.lastRefLocs.length < 2 && !g.lastRefLocs.includes(refSite.location)) g.lastRefLocs.push(refSite.location);
			}
			pkgCounts.set(site.pkg, (pkgCounts.get(site.pkg) ?? 0) + 1);
		}
		_leakTrackingMap.clear();
		_lastRefMap.clear();
		const sorted = [...groups.entries()].sort((a, b) => Number(a[1].pkg !== null) - Number(b[1].pkg !== null) || b[1].count - a[1].count);
		const total = [...pkgCounts.values()].reduce((a, b) => a + b, 0);
		const lines = [`${leaked} slot(s) leaked (${total} tracked array(s)):\n`];
		for (const [loc, g] of sorted) {
			const descs = g.descs.join(", ") + (g.count > g.descs.length ? ", …" : "");
			const tagParts = [];
			if (g.pkg) tagParts.push(g.pkg);
			if (g.via) tagParts.push(`via ${g.via}`);
			const tag = tagParts.length > 0 ? `  (${tagParts.join(", ")})` : "";
			lines.push(`  ${loc} — ${g.count}× ${descs}${tag}`);
			for (const refLoc of g.lastRefLocs) lines.push(`    ↳ last .ref at ${refLoc}`);
		}
		const summaryParts = [];
		for (const [pkg, count] of pkgCounts) summaryParts.push(pkg ? `${count} in ${pkg}` : `${count} from user code`);
		if (summaryParts.length > 0) lines.push("", summaryParts.join(", ") + ".");
		const tips = [];
		if (leaked > total) tips.push(`${leaked - total} slot(s) not in tracked list (created before start() or by internal buffers).`);
		if (total > leaked) tips.push(`Some tracked arrays share backend storage (${total} tracked, ${leaked} slots).`);
		if (sorted.some(([, g]) => !g.pkg && g.maxRc === 1)) tips.push("rc=1 → never disposed. Call .dispose() when done.");
		if (sorted.some(([, g]) => g.maxRc >= 2)) tips.push("rc≥2 → extra .ref without matching .dispose().");
		if (pkgCounts.size > (pkgCounts.has(null) ? 1 : 0)) tips.push("Package-tagged leaks are library bugs, not user error.");
		if (tips.length === 0) tips.push("Call .dispose() on unused results.");
		lines.push("", ...tips);
		const summary = lines.join("\n");
		return {
			leaked,
			details,
			summary
		};
	},
	get active() {
		return _active;
	}
};

//#endregion
//#region src/frontend/convolution.ts
/**
* Check that the shapes and parameters passed to convolution are valid.
* Expected shapes of the lhs and rhs of the convolution are:
*
* - `lhsShape = [*vmapDims, batchSize, inChannels, spatialDims...]`
* - `rhsShape = [*vmapDims, outChannels, inChannels, kernelSize...]`
*
* If the check succeeds, returns the output shape.
*/
function checkConvShape(lhsShape, rhsShape, { vmapDims, strides, padding, lhsDilation, rhsDilation }) {
	if (lhsShape.length !== rhsShape.length) throw new Error(`conv() requires inputs with the same number of dimensions, got ${lhsShape.length} and ${rhsShape.length}`);
	const n = lhsShape.length - 2 - vmapDims;
	if (n < 0) throw new Error("conv() requires at least 2D inputs");
	if (strides.length !== n) throw new Error("conv() strides != spatial dims");
	if (padding.length !== n) throw new Error("conv() padding != spatial dims");
	if (lhsDilation.length !== n) throw new Error("conv() lhsDilation != spatial dimensions");
	if (rhsDilation.length !== n) throw new Error("conv() rhsDilation != spatial dimensions");
	if (lhsShape[vmapDims + 1] !== rhsShape[vmapDims + 1]) throw new Error(`conv() input channels: ${lhsShape[1]} != ${rhsShape[1]}`);
	const outShape = [
		...require_backend.generalBroadcast(lhsShape.slice(0, vmapDims), rhsShape.slice(0, vmapDims)),
		lhsShape[vmapDims],
		rhsShape[vmapDims]
	];
	for (let i = 0; i < n; i++) {
		if (strides[i] <= 0 || !Number.isInteger(strides[i])) throw new Error(`conv() strides[${i}] must be a positive integer`);
		if (padding[i].length !== 2 || !padding[i].every(Number.isInteger)) throw new Error(`conv() padding[${i}] must be a 2-tuple of integers`);
		if (lhsDilation[i] <= 0 || !Number.isInteger(lhsDilation[i])) throw new Error(`conv() lhsDilation[${i}] must be a positive integer`);
		if (rhsDilation[i] <= 0 || !Number.isInteger(rhsDilation[i])) throw new Error(`conv() rhsDilation[${i}] must be a positive integer`);
		const [x, k] = [lhsShape[i + vmapDims + 2], rhsShape[i + vmapDims + 2]];
		if (k <= 0) throw new Error("conv() kernel size must be positive");
		const [pl, pr] = padding[i];
		if (pl < -x || pr < -x || pl + pr < -x) throw new Error(`conv() padding[${i}]=(${pl},${pr}) is too negative for input size ${x}`);
		const kernelSize = (k - 1) * rhsDilation[i] + 1;
		const inSize = Math.max((x - 1) * lhsDilation[i] + 1, 0) + pl + pr;
		if (kernelSize > inSize) throw new Error(`conv() kernel size ${kernelSize} > input size ${inSize} in dimension ${i}`);
		outShape.push(Math.ceil((inSize - kernelSize + 1) / strides[i]));
	}
	return outShape;
}
function checkPoolShape(inShape, window, strides) {
	if (strides.length !== window.length) throw new Error("pool() strides != window dims");
	if (window.length > inShape.length) throw new Error("pool() window has more dimensions than input");
	const outShape = inShape.slice(0, inShape.length - window.length);
	for (let i = 0; i < window.length; i++) {
		const k = window[i];
		const s = strides[i];
		const size$1 = inShape[inShape.length - window.length + i];
		if (k <= 0 || !Number.isInteger(k)) throw new Error(`pool() window[${i}] must be a positive integer`);
		if (k > size$1) throw new Error(`pool() window[${i}]=${k} > input size ${size$1}`);
		if (s <= 0 || !Number.isInteger(s)) throw new Error(`pool() strides[${i}] must be a positive integer`);
		outShape.push(Math.ceil((size$1 - k + 1) / s));
	}
	return outShape.concat(window);
}
/**
* Takes a shape tracker and a kernel size `ks`, then reshapes it so the last
* `ks.length` dimensions become `2 * ks.length` dimensions by treating them as
* spatial dimensions convolved with a kernel.
*
* The resulting array can be multiplied with a kernel of shape `ks`, then
* reduced along the last `ks.length` dimensions for a convolution.
*
* Reference: https://github.com/tinygrad/tinygrad/blob/v0.10.3/tinygrad/tensor.py#L2097
*/
function pool(st, ks, strides = 1, dilation = 1) {
	if (ks.length === 0) return st;
	if (st.shape.length < ks.length) throw new Error("pool() called with too many dimensions");
	if (typeof strides === "number") strides = require_backend.rep(ks.length, strides);
	if (typeof dilation === "number") dilation = require_backend.rep(ks.length, dilation);
	if (strides.some((s) => s <= 0 || !Number.isInteger(s))) throw new Error("pool() strides must be positive integers");
	if (dilation.some((d) => d <= 0 || !Number.isInteger(d))) throw new Error("pool() dilation must be positive integers");
	const noop = st.shape.slice(0, -ks.length);
	const i_ = st.shape.slice(-ks.length);
	const s_ = strides;
	const d_ = dilation;
	const o_ = require_backend.zipn(i_, d_, ks, s_).map(([i, d, k, s]) => Math.ceil((i - d * (k - 1)) / s));
	if (d_.every((d) => d === 1) && ks.every((k, j) => k <= s_[j])) {
		st = st.padOrShrink([...noop.map(() => [0, 0]), ...require_backend.zipn(i_, o_, s_).map(([i, o, s]) => [0, o * s - i])]);
		st = st.reshape([...noop, ...require_backend.zip(o_, s_).flatMap(([o, s]) => [o, s])]).shrink([...noop.map((x) => [0, x]), ...require_backend.zip(o_, ks).flatMap(([o, k]) => [[0, o], [0, k]])]);
		st = st.permute([
			...require_backend.range(noop.length),
			...ks.map((_, j) => noop.length + 2 * j),
			...ks.map((_, j) => noop.length + 2 * j + 1)
		]);
		return st;
	}
	const f_ = require_backend.zipn(o_, s_, i_, d_, ks).map(([o, s, i, d, k]) => 1 + Number(o * s > i - d * (k - 1)));
	const kidf = require_backend.zipn(ks, i_, d_, f_);
	st = st.repeat([...require_backend.rep(noop.length, 1), ...kidf.map(([k, i, d, f]) => Math.ceil(k * (i * f + d) / i))]);
	st = st.shrink([...noop.map((x) => [0, x]), ...kidf.map(([k, i, d, f]) => [0, k * (i * f + d)])]).reshape([...noop, ...kidf.flatMap(([k, i, d, f]) => [k, i * f + d])]);
	const kos = require_backend.zipn(ks, o_, s_);
	st = st.shrink([...noop.map((x) => [0, x]), ...kos.flatMap(([k, o, s]) => [[0, k], [0, o * s]])]).reshape([...noop, ...kos.flat(1)]);
	st = st.shrink([...noop.map((x) => [0, x]), ...kos.flatMap(([k, o]) => [
		[0, k],
		[0, o],
		[0, 1]
	])]).reshape([...noop, ...kos.flatMap(([k, o]) => [k, o])]);
	st = st.permute([
		...require_backend.range(noop.length),
		...ks.map((_, j) => noop.length + 2 * j + 1),
		...ks.map((_, j) => noop.length + 2 * j)
	]);
	return st;
}
/**
* Perform the transpose of pool, directly undo-ing a pool() operation.
*
* Note that since pool repeats the input, the transpose operation technically
* should include a sum reduction. This function doesn't perform the reduction,
* which should be done on the last `k` axes of the returned shape.
*/
function poolTranspose(st, inShape, ks, strides = 1, dilation = 1) {
	if (ks.length === 0) return st;
	if (typeof strides === "number") strides = require_backend.rep(ks.length, strides);
	if (typeof dilation === "number") dilation = require_backend.rep(ks.length, dilation);
	const noop = inShape.slice(0, -ks.length);
	const i_ = inShape.slice(-ks.length);
	const s_ = strides;
	const d_ = dilation;
	const o_ = require_backend.zipn(i_, d_, ks, s_).map(([i, d, k, s]) => Math.ceil((i - d * (k - 1)) / s));
	if (d_.every((d) => d === 1) && ks.every((k, j) => k <= s_[j])) {
		st = st.permute([...require_backend.range(noop.length), ...ks.flatMap((_, j) => [noop.length + j, noop.length + o_.length + j])]);
		st = st.pad([...noop.map(() => [0, 0]), ...require_backend.zip(s_, ks).flatMap(([s, k]) => [[0, 0], [0, s - k]])]).reshape([...noop, ...require_backend.zip(o_, s_).map(([o, s]) => o * s)]);
		st = st.padOrShrink([...noop.map(() => [0, 0]), ...require_backend.zipn(i_, o_, s_).map(([i, o, s]) => [0, i - o * s])]);
		return st.reshape(st.shape.concat(require_backend.rep(ks.length, 1)));
	}
	if (!require_backend.deepEqual(o_, st.shape.slice(noop.length, noop.length + ks.length))) throw new Error("poolTranspose() called with mismatched output shape");
	const f_ = require_backend.zipn(o_, s_, i_, d_, ks).map(([o, s, i, d, k]) => 1 + Number(o * s > i - d * (k - 1)));
	const kidf = require_backend.zipn(ks, i_, d_, f_);
	const kos = require_backend.zipn(ks, o_, s_);
	st = st.permute([...require_backend.range(noop.length), ...ks.flatMap((_, j) => [noop.length + ks.length + j, noop.length + j])]);
	st = st.reshape([...noop, ...kos.flatMap(([k, o]) => [
		k,
		o,
		1
	])]).pad([...noop.map(() => [0, 0]), ...s_.flatMap((s) => [
		[0, 0],
		[0, 0],
		[0, s - 1]
	])]);
	st = st.reshape([...noop, ...kos.flatMap(([k, o, s]) => [k, o * s])]).pad([...noop.map(() => [0, 0]), ...kidf.flatMap(([_k, i, d, f], j) => [[0, 0], [0, i * f + d - o_[j] * s_[j]]])]);
	st = st.reshape([...noop, ...kidf.map(([k, i, d, f]) => k * (i * f + d))]).pad([...noop.map(() => [0, 0]), ...kidf.map(([k, i, d, f]) => [0, Math.ceil(k * (i * f + d) / i) * i - k * (i * f + d)])]);
	st = st.reshape([...noop, ...kidf.flatMap(([k, i, d, f]) => [Math.ceil(k * (i * f + d) / i), i])]).permute([
		...require_backend.range(noop.length),
		...ks.map((_, j) => noop.length + 2 * j + 1),
		...ks.map((_, j) => noop.length + 2 * j)
	]);
	return st;
}
/** Applies dilation to an array directly, for transposed convolution. */
function applyDilation(st, dilation) {
	if (dilation.every((s) => s === 1)) return st;
	const s_ = dilation;
	const n = s_.length;
	const prefix = st.shape.slice(0, -n);
	const k_ = st.shape.slice(-n);
	st = st.reshape([...prefix, ...k_.flatMap((k) => [k, 1])]);
	st = st.pad([...prefix.map(() => [0, 0]), ...s_.flatMap((s) => [[0, 0], [0, s - 1]])]);
	st = st.reshape([...prefix, ...k_.map((k, i) => k * s_[i])]);
	st = st.shrink([...prefix.map((p) => [0, p]), ...k_.map((k, i) => [0, (k - 1) * s_[i] + 1])]);
	return st;
}
/**
* Prepare for a convolution between two arrays.
*
* This does not check the validity of the shapes, which should be checked
* beforehand using `checkConvShape()`.
*/
function prepareConv(stX, stY, params) {
	const v = params.vmapDims;
	const n = stX.shape.length - 2 - v;
	const vmapShape = stX.shape.slice(0, v);
	stX = applyDilation(stX, params.lhsDilation);
	const ks = stY.shape.slice(v + 2);
	stX = stX.padOrShrink([...require_backend.rep(v + 2, [0, 0]), ...params.padding]);
	stX = pool(stX, ks, params.strides, params.rhsDilation);
	stX = stX.moveaxis(v + 1, v + n + 1).reshape([
		...vmapShape,
		stX.shape[v],
		1,
		...stX.shape.slice(v + 2, v + n + 2),
		stX.shape[v + 1] * require_backend.prod(ks)
	]);
	stY = stY.reshape([
		...vmapShape,
		1,
		stY.shape[v],
		...require_backend.rep(n, 1),
		stY.shape[v + 1] * require_backend.prod(ks)
	]);
	return [stX, stY];
}

//#endregion
//#region src/tree.ts
var tree_exports = {};
__export(tree_exports, {
	JsTreeDef: () => JsTreeDef,
	NodeType: () => NodeType,
	dispose: () => dispose,
	flatten: () => flatten,
	leaves: () => leaves,
	map: () => map,
	ref: () => ref,
	structure: () => structure,
	unflatten: () => unflatten
});
const JsArray$2 = globalThis.Array;
let NodeType = /* @__PURE__ */ function(NodeType$1) {
	NodeType$1["Array"] = "Array";
	NodeType$1["Object"] = "Object";
	NodeType$1["Leaf"] = "Leaf";
	NodeType$1["None"] = "None";
	return NodeType$1;
}({});
/** Represents the structure of a JsTree. */
var JsTreeDef = class JsTreeDef {
	static leaf = new JsTreeDef(NodeType.Leaf, null, []);
	static none = new JsTreeDef(NodeType.None, null, []);
	constructor(nodeType, nodeMetadata, childTreedefs) {
		this.nodeType = nodeType;
		this.nodeMetadata = nodeMetadata;
		this.childTreedefs = childTreedefs;
	}
	/** Get the total number of leaves in the tree. */
	get size() {
		if (this.nodeType === NodeType.None) return 0;
		return this.nodeType === NodeType.Leaf ? 1 : this.childTreedefs.reduce((a, b) => a + b.size, 0);
	}
	/** Returns a string representation of this tree definition. */
	toString(root = true) {
		if (root) return "JsTreeDef(" + this.toString(false) + ")";
		switch (this.nodeType) {
			case NodeType.Leaf: return "*";
			case NodeType.None: return "null";
			case NodeType.Array: return `[${this.childTreedefs.map((x) => x.toString(false)).join(", ")}]`;
			case NodeType.Object: {
				const parts = [];
				for (let i = 0; i < this.childTreedefs.length; i++) parts.push(`${quoteObjectKey(this.nodeMetadata[i])}: ${this.childTreedefs[i].toString(false)}`);
				return `{${parts.join(", ")}}`;
			}
		}
	}
	/** Compare this tree definition with another. */
	equals(other) {
		return this.nodeType === other.nodeType && require_backend.deepEqual(this.nodeMetadata, other.nodeMetadata) && this.childTreedefs.length === other.childTreedefs.length && this.childTreedefs.every((x, i) => x.equals(other.childTreedefs[i]));
	}
};
function quoteObjectKey(key$1) {
	if (/^[a-zA-Z_$][a-zA-Z0-9_$]*$/.test(key$1)) return key$1;
	return JSON.stringify(key$1);
}
/** Flatten a structured object, returning the tree definition. */
function flatten(tree) {
	const leaves$1 = [];
	const treedef = _flatten(tree, leaves$1);
	return [leaves$1, treedef];
}
function _flatten(tree, leaves$1) {
	if (tree === null || tree === void 0) return JsTreeDef.none;
	if (JsArray$2.isArray(tree)) {
		const childTrees = tree.map((c) => _flatten(c, leaves$1));
		return new JsTreeDef(NodeType.Array, null, childTrees);
	} else if (typeof tree === "object" && tree !== null && tree.constructor === Object) {
		const [keys, values] = require_backend.unzip2(Object.entries(tree));
		const childTrees = values.map((c) => _flatten(c, leaves$1));
		return new JsTreeDef(NodeType.Object, keys, childTrees);
	} else {
		leaves$1.push(tree);
		return JsTreeDef.leaf;
	}
}
/** Get the leaves of a tree. */
function leaves(tree) {
	return flatten(tree)[0];
}
/** Get the treedef for a tree. */
function structure(tree) {
	return flatten(tree)[1];
}
/** Reconstruct a structured object from the flattened representation. */
function unflatten(treedef, leaves$1) {
	return _unflatten(treedef, leaves$1[Symbol.iterator]());
}
function _unflatten(treedef, leaves$1) {
	switch (treedef.nodeType) {
		case NodeType.None: return null;
		case NodeType.Leaf: {
			const { value, done } = leaves$1.next();
			if (done) throw new TypeError("Ran out of leaves while unflattening JsTree");
			return value;
		}
		case NodeType.Array: return treedef.childTreedefs.map((c) => _unflatten(c, leaves$1));
		case NodeType.Object: {
			const obj = {};
			for (let i = 0; i < treedef.childTreedefs.length; i++) obj[treedef.nodeMetadata[i]] = _unflatten(treedef.childTreedefs[i], leaves$1);
			return obj;
		}
	}
}
/** Maps a multi-input function over pytree args to produce a new pytree. */
function map(fn, tree, ...rest) {
	const [leaves$1, treedef] = flatten(tree);
	const restLeaves = rest.map((x) => flatten(x)[0]);
	const resultLeaves = [];
	for (let i = 0; i < leaves$1.length; i++) resultLeaves.push(fn(leaves$1[i], ...restLeaves.map((x) => x[i])));
	return unflatten(treedef, resultLeaves);
}
/** Take a reference of every array in a tree. */
function ref(tree) {
	return map((x) => x instanceof Tracer ? x.ref : x, tree);
}
/** Dispose every array in a tree. */
function dispose(tree) {
	if (tree) map((x) => x instanceof Tracer ? x.dispose() : void 0, tree);
}

//#endregion
//#region node_modules/.pnpm/@oxc-project+runtime@0.78.0/node_modules/@oxc-project/runtime/src/helpers/usingCtx.js
var require_usingCtx = /* @__PURE__ */ __commonJS({ "node_modules/.pnpm/@oxc-project+runtime@0.78.0/node_modules/@oxc-project/runtime/src/helpers/usingCtx.js": ((exports, module) => {
	function _usingCtx() {
		var r = "function" == typeof SuppressedError ? SuppressedError : function(r$1, e$2) {
			var n$1 = Error();
			return n$1.name = "SuppressedError", n$1.error = r$1, n$1.suppressed = e$2, n$1;
		}, e$1 = {}, n = [];
		function using(r$1, e$2) {
			if (null != e$2) {
				if (Object(e$2) !== e$2) throw new TypeError("using declarations can only be used with objects, functions, null, or undefined.");
				if (r$1) var o = e$2[Symbol.asyncDispose || Symbol["for"]("Symbol.asyncDispose")];
				if (void 0 === o && (o = e$2[Symbol.dispose || Symbol["for"]("Symbol.dispose")], r$1)) var t = o;
				if ("function" != typeof o) throw new TypeError("Object is not disposable.");
				t && (o = function o$1() {
					try {
						t.call(e$2);
					} catch (r$2) {
						return Promise.reject(r$2);
					}
				}), n.push({
					v: e$2,
					d: o,
					a: r$1
				});
			} else r$1 && n.push({
				d: e$2,
				a: r$1
			});
			return e$2;
		}
		return {
			e: e$1,
			u: using.bind(null, !1),
			a: using.bind(null, !0),
			d: function d() {
				var o, t = this.e, s = 0;
				function next() {
					for (; o = n.pop();) try {
						if (!o.a && 1 === s) return s = 0, n.push(o), Promise.resolve().then(next);
						if (o.d) {
							var r$1 = o.d.call(o.v);
							if (o.a) return s |= 2, Promise.resolve(r$1).then(next, err);
						} else s |= 1;
					} catch (r$2) {
						return err(r$2);
					}
					if (1 === s) return t !== e$1 ? Promise.reject(t) : Promise.resolve();
					if (t !== e$1) throw t;
				}
				function err(n$1) {
					return t = t !== e$1 ? new r(n$1, t) : n$1, next();
				}
				return next();
			}
		};
	}
	module.exports = _usingCtx, module.exports.__esModule = true, module.exports["default"] = module.exports;
}) });

//#endregion
//#region src/frontend/core.ts
var import_usingCtx$12 = /* @__PURE__ */ __toESM(require_usingCtx(), 1);
/**
* Frontend primitive operations, which are lowered into Kernel objects before
* being dispatched to the backend.
*
* Any operation between arrays can be described in these parts. This is also
* the set of primitives that can occur in Jaxpr programs, and the level at
* which transformations like vmap, grad, and jvp occur. They are loosely based
* on [XLA](https://openxla.org/xla/operation_semantics).
*
* All n-ary operations support broadcasting, with NumPy semantics.
*/
let Primitive = /* @__PURE__ */ function(Primitive$1) {
	Primitive$1["Add"] = "add";
	Primitive$1["Mul"] = "mul";
	Primitive$1["Idiv"] = "idiv";
	Primitive$1["Mod"] = "mod";
	Primitive$1["Min"] = "min";
	Primitive$1["Max"] = "max";
	Primitive$1["Neg"] = "neg";
	Primitive$1["Reciprocal"] = "reciprocal";
	Primitive$1["Floor"] = "floor";
	Primitive$1["Ceil"] = "ceil";
	Primitive$1["StopGradient"] = "stop_gradient";
	Primitive$1["Cast"] = "cast";
	Primitive$1["Bitcast"] = "bitcast";
	Primitive$1["Sin"] = "sin";
	Primitive$1["Cos"] = "cos";
	Primitive$1["Asin"] = "asin";
	Primitive$1["Atan"] = "atan";
	Primitive$1["Exp"] = "exp";
	Primitive$1["Log"] = "log";
	Primitive$1["Erf"] = "erf";
	Primitive$1["Erfc"] = "erfc";
	Primitive$1["Sqrt"] = "sqrt";
	Primitive$1["Reduce"] = "reduce";
	Primitive$1["Dot"] = "dot";
	Primitive$1["Conv"] = "conv";
	Primitive$1["Pool"] = "pool";
	Primitive$1["PoolTranspose"] = "pool_transpose";
	Primitive$1["Compare"] = "compare";
	Primitive$1["Where"] = "where";
	Primitive$1["Concatenate"] = "concatenate";
	Primitive$1["Split"] = "split";
	Primitive$1["RandomBits"] = "random_bits";
	Primitive$1["Gather"] = "gather";
	Primitive$1["Transpose"] = "transpose";
	Primitive$1["Broadcast"] = "broadcast";
	Primitive$1["Reshape"] = "reshape";
	Primitive$1["Flip"] = "flip";
	Primitive$1["Shrink"] = "shrink";
	Primitive$1["Pad"] = "pad";
	Primitive$1["DynamicUpdateSlice"] = "dynamic_update_slice";
	Primitive$1["Sort"] = "sort";
	Primitive$1["Argsort"] = "argsort";
	Primitive$1["TriangularSolve"] = "triangular_solve";
	Primitive$1["Cholesky"] = "cholesky";
	Primitive$1["LU"] = "lu";
	Primitive$1["Jit"] = "jit";
	Primitive$1["Scan"] = "scan";
	return Primitive$1;
}({});
let CompareOp = /* @__PURE__ */ function(CompareOp$1) {
	CompareOp$1["Less"] = "less";
	CompareOp$1["Equal"] = "equal";
	CompareOp$1["NotEqual"] = "not_equal";
	CompareOp$1["LessEqual"] = "less_equal";
	return CompareOp$1;
}({});
const routinePrimitives = new Map([
	[Primitive.Sort, require_backend.Routines.Sort],
	[Primitive.Argsort, require_backend.Routines.Argsort],
	[Primitive.TriangularSolve, require_backend.Routines.TriangularSolve],
	[Primitive.Cholesky, require_backend.Routines.Cholesky],
	[Primitive.LU, require_backend.Routines.LU]
]);
function add$1(x, y) {
	return bind1(Primitive.Add, [x, y]);
}
function mul(x, y) {
	return bind1(Primitive.Mul, [x, y]);
}
function idiv(x, y) {
	return bind1(Primitive.Idiv, [x, y]);
}
function mod(x, y) {
	return bind1(Primitive.Mod, [x, y]);
}
function min$1(x, y) {
	return bind1(Primitive.Min, [x, y]);
}
function max$1(x, y) {
	return bind1(Primitive.Max, [x, y]);
}
function neg(x) {
	return bind1(Primitive.Neg, [x]);
}
function reciprocal$1(x) {
	return bind1(Primitive.Reciprocal, [x]);
}
function floor$1(x) {
	return bind1(Primitive.Floor, [x]);
}
function ceil$1(x) {
	return bind1(Primitive.Ceil, [x]);
}
function stopGradient(x) {
	return bind1(Primitive.StopGradient, [x]);
}
function cast(x, dtype) {
	return bind1(Primitive.Cast, [x], { dtype });
}
function bitcast(x, dtype) {
	return bind1(Primitive.Bitcast, [x], { dtype });
}
function sin$1(x) {
	return bind1(Primitive.Sin, [x]);
}
function cos$1(x) {
	return bind1(Primitive.Cos, [x]);
}
function asin$1(x) {
	return bind1(Primitive.Asin, [x]);
}
function atan$1(x) {
	return bind1(Primitive.Atan, [x]);
}
function exp$1(x) {
	return bind1(Primitive.Exp, [x]);
}
function log$1(x) {
	return bind1(Primitive.Log, [x]);
}
function erf$1(x) {
	return bind1(Primitive.Erf, [x]);
}
function erfc$1(x) {
	return bind1(Primitive.Erfc, [x]);
}
function sqrt$1(x) {
	return bind1(Primitive.Sqrt, [x]);
}
function reduce(x, op, axis = null, opts) {
	if (!require_backend.AluGroup.Reduce.has(op)) throw new TypeError(`Invalid reduce operation: ${op}`);
	axis = require_backend.normalizeAxis(axis, ndim$1(x));
	const originalShape = getShape(x);
	let result = bind1(Primitive.Reduce, [x], {
		op,
		axis
	});
	if (opts?.keepdims) {
		const preReshape = result;
		result = result.reshape(originalShape.map((dim, i) => axis.includes(i) ? 1 : dim));
		if (preReshape !== result) preReshape.dispose();
	}
	return result;
}
function dot$2(x, y) {
	return bind1(Primitive.Dot, [x, y]);
}
function conv$1(x, y, params = {}) {
	if (x.ndim !== y.ndim) throw new Error(`conv() requires inputs with the same number of dimensions, got ${x.ndim} and ${y.ndim}`);
	const vmapDims = params.vmapDims ?? 0;
	const n = x.ndim - 2 - vmapDims;
	if (n < 0) throw new Error("conv() requires at least 2D inputs");
	return bind1(Primitive.Conv, [x, y], {
		vmapDims,
		strides: params.strides ?? require_backend.rep(n, 1),
		padding: params.padding ?? require_backend.rep(n, [0, 0]),
		lhsDilation: params.lhsDilation ?? require_backend.rep(n, 1),
		rhsDilation: params.rhsDilation ?? require_backend.rep(n, 1)
	});
}
function compare(x, y, op) {
	return bind1(Primitive.Compare, [x, y], { op });
}
function greater$1(x, y) {
	return compare(y, x, CompareOp.Less);
}
function less$1(x, y) {
	return compare(x, y, CompareOp.Less);
}
function equal$1(x, y) {
	return compare(x, y, CompareOp.Equal);
}
function notEqual$1(x, y) {
	return compare(x, y, CompareOp.NotEqual);
}
function greaterEqual$1(x, y) {
	return compare(y, x, CompareOp.LessEqual);
}
function lessEqual$1(x, y) {
	return compare(x, y, CompareOp.LessEqual);
}
function where$1(cond, x, y) {
	return bind1(Primitive.Where, [
		cond,
		x,
		y
	]);
}
function concatenate$1(xs, axis) {
	if (xs.length === 0) throw new Error("concatenate requires at least one input");
	const avals = xs.map((x) => ShapedArray.fromAval(getAval(x)));
	axis = require_backend.checkAxis(axis, avals[0].ndim);
	for (const x of avals) if (x.ndim !== avals[0].ndim || !x.shape.every((s, i) => i === axis || s === avals[0].shape[i])) throw new Error(`Concatenate: inputs ${avals[0]} and ${x} must match shapes except on axis ${axis}`);
	const MAX_CONCAT_CHUNK = 7;
	if (xs.length > MAX_CONCAT_CHUNK) {
		const chunks = [];
		for (let i = 0; i < xs.length; i += MAX_CONCAT_CHUNK) {
			const group = xs.slice(i, Math.min(i + MAX_CONCAT_CHUNK, xs.length));
			chunks.push(group.length === 1 ? group[0] : bind1(Primitive.Concatenate, group, { axis }));
		}
		const result = concatenate$1(chunks, axis);
		if (!insideAbstractTrace()) {
			const xsSet = new Set(xs);
			for (const c of chunks) if (c !== result && !xsSet.has(c) && typeof c.dispose === "function") c.dispose();
		}
		return result;
	}
	return bind1(Primitive.Concatenate, xs, { axis });
}
function split$2(x, axis, sizes) {
	axis = require_backend.checkAxis(axis, ndim$1(x));
	if (sizes.some((s) => s < 0 || !Number.isInteger(s))) throw new Error(`split: sizes must be nonnegative integers, got ${JSON.stringify(sizes)}`);
	const totalSize = sizes.reduce((a, b) => a + b, 0);
	if (totalSize !== getShape(x)[axis]) throw new Error(`split: sizes must sum to the size of the axis ${axis}, got ${totalSize}`);
	return bind(Primitive.Split, [x], {
		axis,
		sizes
	});
}
function randomBits(k0, k1, shape$1, mode = "xor") {
	if (!require_backend.deepEqual(k0.shape, k1.shape) || k0.dtype !== require_backend.DType.Uint32 || k1.dtype !== require_backend.DType.Uint32) throw new Error(`randomBits: key parts must be uint32 with the same shape, got ${ShapedArray.fromAval(k0.aval)} and ${ShapedArray.fromAval(k1.aval)}`);
	return bind1(Primitive.RandomBits, [k0, k1], {
		shape: shape$1,
		mode
	});
}
function gather(x, indices, axis, outDim) {
	if (indices.length === 0) throw new Error("gather() requires at least one index");
	if (!Array.isArray(axis) || axis.length !== indices.length) throw new Error(`Invalid gather() axis: expected ${indices.length} axes, got ${JSON.stringify(axis)}`);
	axis = axis.map((a) => require_backend.checkAxis(a, ndim$1(x)));
	if (new Set(axis).size !== axis.length) throw new Error(`Invalid gather() axis: duplicate axes ${JSON.stringify(axis)}`);
	outDim = require_backend.checkAxis(outDim, ndim$1(x) - axis.length + 1);
	return bind1(Primitive.Gather, [x, ...indices], {
		axis,
		outDim
	});
}
function transpose$1(x, perm) {
	perm = perm ? perm.map((a) => require_backend.checkAxis(a, ndim$1(x))) : require_backend.range(ndim$1(x)).reverse();
	if (!require_backend.isPermutation(perm, ndim$1(x))) throw new Error(`Invalid transpose permutation for ${ndim$1(x)} axes: ${JSON.stringify(perm)}`);
	return bind1(Primitive.Transpose, [x], { perm });
}
function broadcast(x, shape$1, axis) {
	axis = require_backend.normalizeAxis(axis, shape$1.length);
	return bind1(Primitive.Broadcast, [x], {
		shape: shape$1,
		axis
	});
}
function reshape$1(x, shape$1) {
	if (typeof shape$1 === "number") shape$1 = [shape$1];
	const originalShape = getShape(x);
	const autoIdx = shape$1.indexOf(-1);
	if (autoIdx !== -1) {
		const remaining = require_backend.prod(originalShape) / -require_backend.prod(shape$1);
		if (!Number.isInteger(remaining) || remaining < 0) throw new Error(`Invalid reshape: ${JSON.stringify(originalShape)} -> ${JSON.stringify(shape$1)}`);
		shape$1 = shape$1.toSpliced(autoIdx, 1, remaining);
	}
	if (require_backend.prod(originalShape) !== require_backend.prod(shape$1)) throw new Error(`Invalid reshape: ${JSON.stringify(originalShape)} -> ${JSON.stringify(shape$1)}`);
	return bind1(Primitive.Reshape, [x], { shape: shape$1 });
}
function flip$1(x, axis) {
	axis = require_backend.normalizeAxis(axis, ndim$1(x));
	return bind1(Primitive.Flip, [x], { axis });
}
function shrink(x, slice) {
	const shape$1 = getShape(x);
	if (!Array.isArray(slice) || !slice.every(require_backend.isNumberPair)) throw new Error(`Invalid shrink() type: ${JSON.stringify(slice)}`);
	if (slice.length !== shape$1.length) throw new Error(`Invalid shrink(): expected ${shape$1.length} axes, got ${slice.length}`);
	for (let i = 0; i < shape$1.length; i++) {
		const [start, end] = slice[i];
		if (start > end || start < 0 || end > shape$1[i]) throw new Error(`Invalid shrink() slice for axis ${i}: [${start}, ${end}] on shape ${shape$1[i]}`);
	}
	return bind1(Primitive.Shrink, [x], { slice });
}
function pad$1(x, width) {
	const nd = ndim$1(x);
	let w;
	if (typeof width === "number") w = [[width, width]];
	else if (require_backend.isNumberPair(width)) w = [width];
	else if (!Array.isArray(width)) {
		const indicesAndPairs = Object.entries(width);
		w = require_backend.rep(nd, [0, 0]);
		for (const [k, v] of indicesAndPairs) w[require_backend.checkAxis(parseInt(k), nd)] = v;
	} else if (!width.every(require_backend.isNumberPair)) throw new TypeError(`Invalid pad() type: ${JSON.stringify(width)}`);
	else w = width;
	if (w.length === 1) {
		const [w0, w1] = w[0];
		w = require_backend.rep(nd, () => [w0, w1]);
	} else if (w.length !== nd) throw new Error(`Invalid pad(): expected ${nd} axes, got ${w.length}`);
	return bind1(Primitive.Pad, [x], { width: w });
}
function triangularSolve$1(a, b, { lower = false, unitDiagonal = false } = {}) {
	try {
		var _usingCtx$1 = (0, import_usingCtx$12.default)();
		const as = getShape(a);
		const bs = getShape(b);
		if (as.length < 2 || bs.length < 2) throw new Error(`triangular_solve: must be >=2D, got a=${as}, b=${bs}`);
		const n = as[as.length - 2];
		if (n !== as[as.length - 1] || n !== bs[bs.length - 1]) throw new Error(`triangular_solve: incompatible shapes a=${as}, b=${bs}`);
		if (!lower) return bind1(Primitive.TriangularSolve, [a, b], { unitDiagonal });
		const aFlipped = _usingCtx$1.u(flip$1(a, [-2, -1]));
		const bFlipped = _usingCtx$1.u(flip$1(b, [-1]));
		const x = _usingCtx$1.u(bind1(Primitive.TriangularSolve, [aFlipped, bFlipped], { unitDiagonal }));
		return flip$1(x, [-1]);
	} catch (_) {
		_usingCtx$1.e = _;
	} finally {
		_usingCtx$1.d();
	}
}
function cholesky$2(x) {
	const aval = ShapedArray.fromAval(getAval(x));
	if (aval.ndim < 2 || aval.shape[aval.ndim - 1] !== aval.shape[aval.ndim - 2]) throw new Error(`cholesky: expected batch of square matrices, got ${aval}`);
	return bind1(Primitive.Cholesky, [x]);
}
function lu$1(x) {
	const aval = ShapedArray.fromAval(getAval(x));
	if (aval.ndim < 2) throw new Error(`lu: expected batch of matrices, got ${aval}`);
	return bind(Primitive.LU, [x]);
}
function sort$1(x) {
	const nd = ndim$1(x);
	if (nd === 0) throw new Error("sort: requires at least 1D input");
	return bind1(Primitive.Sort, [x]);
}
function argsort$1(x) {
	const nd = ndim$1(x);
	if (nd === 0) throw new Error("argsort: requires at least 1D input");
	return bind(Primitive.Argsort, [x]);
}
function bind1(prim, args, params = {}) {
	const [results] = bind(prim, args, params);
	return results;
}
const traceStack = [];
let dynamicTrace = null;
/**
* Push an interpreter onto the trace stack. Use this like:
* `using main = newMain(...);`
*/
function newMain(traceType, globalData = null) {
	const level = traceStack.length;
	const main = {
		level,
		traceType,
		globalData
	};
	traceStack.push(main);
	return Object.assign(main, { [Symbol.dispose]() {
		traceStack.pop();
	} });
}
/**
* Set the current dynamic trace, which stashes the current interpreter stack
* and acts temporarily as the bottom of the stack. Use this like:
* `using _dynamic = newDynamic(main);`
*/
function newDynamic(main) {
	const prevDynamicTrace = dynamicTrace;
	dynamicTrace = main;
	return { [Symbol.dispose]() {
		dynamicTrace = prevDynamicTrace;
	} };
}
function currentTraceLevel() {
	return traceStack[traceStack.length - 1].level;
}
/**
* Flag set only during scan body tracing. Arrays created while this is true
* are tagged as anonymous builder-owned consts (via anonymousConstArrays).
* Unlike insideTrace(), this doesn't fire during jit/vmap/other traces where
* the ClosedJaxpr is cached and consts must survive beyond compilation.
*/
let _scanBodyTraceActiveFlag = false;
function isScanBodyTraceActive() {
	return _scanBodyTraceActiveFlag;
}
function setScanBodyTraceActive(v) {
	_scanBodyTraceActiveFlag = v;
}
/**
* Returns true if any abstract (graph-building) trace is on the stack.
*
* Unlike `insideTrace()`, this returns false when only concrete-value traces
* like BatchTrace are active. Cleanup of intermediate arrays is safe when
* no abstract traces are present — BatchTrace wraps concrete arrays that are
* independently owned, so disposing them doesn't affect outer trace state.
*
* Use this instead of `insideTrace()` at cleanup sites where the values being
* disposed are concrete arrays (not abstract tracers from a graph-building trace).
*/
function insideAbstractTrace() {
	for (let i = 1; i < traceStack.length; i++) if (traceStack[i].isAbstract) return true;
	return false;
}
/**
* Returns true if any abstract trace exists below the given level on the stack.
*
* Used by jvpFlat to decide whether its own intermediate cleanup is safe.
* When only BatchTraces are below, cleanup is safe. When PE or JaxprTrace
* is below, the intermediates are managed by that outer trace.
*/
function hasAbstractTraceBelow(level) {
	for (let i = 1; i < level && i < traceStack.length; i++) if (traceStack[i].isAbstract) return true;
	return false;
}
var Trace = class {
	constructor(main) {
		this.main = main;
	}
};
/**
* Broadcast shapes and promote types with casting for two avals.
*
* This implements the weak type behavior described in `promoteTypes()`, but not
* implemented in that function as `weakType` is not passed.
*/
function promoteAvals(a, b) {
	const shape$1 = require_backend.generalBroadcast(a.shape, b.shape);
	const weakType = a.weakType && b.weakType;
	let dtype;
	if (a.weakType === b.weakType) dtype = require_backend.promoteTypes(a.dtype, b.dtype);
	else if (a.weakType) dtype = require_backend.promoteTypes(b.dtype, require_backend.DType.Uint32);
	else dtype = require_backend.promoteTypes(a.dtype, require_backend.DType.Uint32);
	return new ShapedArray(shape$1, dtype, weakType);
}
var Tracer = class Tracer {
	/** @ignore */
	_trace;
	constructor(trace$1) {
		this._trace = trace$1;
	}
	/**
	* Enables `using` declarations for automatic disposal at block exit.
	*
	* Default is a no-op on the abstract Tracer base. Only concrete `Array`
	* overrides this to call `dispose()`. Calling `dispose()` here would be
	* unsafe for JVPTracer / BatchTracer / PartialEvalTracer because it would
	* cascade prematurely during `grad()` / `jvp()` / `vmap()` tracing.
	*/
	[Symbol.dispose]() {}
	/** Current reference count (only meaningful on concrete Arrays). */
	get refCount() {
		return 0;
	}
	/** The shape of the array. */
	get shape() {
		return this.aval.shape;
	}
	/** The total number of elements in the array. */
	get size() {
		return require_backend.prod(this.shape);
	}
	/** The dtype of elements stored in the array. */
	get dtype() {
		return this.aval.dtype;
	}
	/**
	* Whether the array is weakly typed.
	*
	* Weakly typed arrays will cast to the dtype of the other operand. See
	* `promoteTypes()` for details.
	*/
	get weakType() {
		return this.aval.weakType;
	}
	/** The number of dimensions of the array. */
	get ndim() {
		return this.shape.length;
	}
	/** @ignore */
	fullLower() {
		return this;
	}
	neg() {
		return neg(this);
	}
	add(other) {
		return add$1(this, other);
	}
	mul(other) {
		return mul(this, other);
	}
	mod(other) {
		return mod(this, other);
	}
	greater(other) {
		return greater$1(this, other);
	}
	less(other) {
		return less$1(this, other);
	}
	equal(other) {
		return equal$1(this, other);
	}
	notEqual(other) {
		return notEqual$1(this, other);
	}
	greaterEqual(other) {
		return greaterEqual$1(this, other);
	}
	lessEqual(other) {
		return lessEqual$1(this, other);
	}
	/** Sum of the elements of the array over a given axis, or axes. */
	sum(axis = null, opts) {
		return reduce(this, require_backend.AluOp.Add, axis, opts);
	}
	/** Product of the array elements over a given axis. */
	prod(axis = null, opts) {
		return reduce(this, require_backend.AluOp.Mul, axis, opts);
	}
	/** Compute the average of the array elements along the specified axis. */
	mean(axis = null, opts) {
		axis = require_backend.normalizeAxis(axis, this.ndim);
		const n = axis.reduce((acc, a) => acc * this.shape[a], 1);
		if (n === 0) throw new Error("mean: cannot compute mean over zero-length axis");
		const originalDtype = this.dtype;
		const castDtype = require_backend.promoteTypes(originalDtype, require_backend.DType.Float32);
		const casted = this.astype(castDtype);
		let summed = null;
		let scaled = null;
		let final_ = null;
		try {
			summed = reduce(casted, require_backend.AluOp.Add, axis, opts);
			scaled = summed.mul(1 / n);
			final_ = scaled.astype(originalDtype);
			return final_;
		} finally {
			if (scaled && scaled !== final_) scaled.dispose();
			if (summed && summed !== scaled && summed !== final_) summed.dispose();
			if (casted !== this && casted !== summed && casted !== scaled && casted !== final_) casted.dispose();
		}
	}
	/** Minimum of the elements of the array along a given axis. */
	min(axis = null, opts) {
		return reduce(this, require_backend.AluOp.Min, axis, opts);
	}
	/** Maximum of the elements of the array along a given axis. */
	max(axis = null, opts) {
		return reduce(this, require_backend.AluOp.Max, axis, opts);
	}
	/** Test whether all array elements along a given axis evaluate to true. */
	all(axis = null, opts) {
		const boolArr = this.astype(require_backend.DType.Bool);
		try {
			return boolArr.min(axis, opts);
		} finally {
			if (boolArr !== this) boolArr.dispose();
		}
	}
	/** Test whether any array element along a given axis evaluates to true. */
	any(axis = null, opts) {
		const boolArr = this.astype(require_backend.DType.Bool);
		try {
			return boolArr.max(axis, opts);
		} finally {
			if (boolArr !== this) boolArr.dispose();
		}
	}
	/** Permute the dimensions of an array. Defaults to reversing the axis order. */
	transpose(perm) {
		return transpose$1(this, perm);
	}
	/**
	* Give a new shape to an array without changing its data.
	*
	* One shape dimension can be -1. In this case, the value is inferred from the
	* length of the array and remaining dimensions.
	*/
	reshape(shape$1) {
		return reshape$1(this, shape$1);
	}
	/** Copy the array and cast to a specified dtype. */
	astype(dtype) {
		if (this.dtype === dtype) return this;
		return cast(this, dtype);
	}
	/** Subtract an array from this one. */
	sub(other) {
		try {
			var _usingCtx3 = (0, import_usingCtx$12.default)();
			const n = _usingCtx3.u(neg(other));
			return this.add(n);
		} catch (_) {
			_usingCtx3.e = _;
		} finally {
			_usingCtx3.d();
		}
	}
	/** Divide an array by this one. */
	div(other) {
		if (require_backend.isFloatDtype(this.dtype)) try {
			var _usingCtx4 = (0, import_usingCtx$12.default)();
			const r = _usingCtx4.u(reciprocal$1(other));
			return this.mul(r);
		} catch (_) {
			_usingCtx4.e = _;
		} finally {
			_usingCtx4.d();
		}
		return idiv(this, other);
	}
	/** Return specified diagonals. See `jax.numpy.diagonal` for full docs. */
	diagonal(offset = 0, axis1 = 0, axis2 = 1) {
		if (!Number.isInteger(offset)) throw new TypeError(`offset must be an integer, got ${offset}`);
		if (offset < 0) return this.diagonal(-offset, axis2, axis1);
		axis1 = require_backend.checkAxis(axis1, this.ndim);
		axis2 = require_backend.checkAxis(axis2, this.ndim);
		if (axis1 === axis2) throw new Error("axis1 and axis2 must not be equal");
		if (offset >= this.shape[axis2]) throw new Error("offset exceeds axis size");
		const toDispose = [];
		let ar = this;
		if (axis1 !== ar.ndim - 2 || axis2 !== ar.ndim - 1) {
			const perm = require_backend.range(ar.ndim).filter((i) => i !== axis1 && i !== axis2).concat(axis1, axis2);
			ar = ar.transpose(perm);
			toDispose.push(ar);
		}
		const [n, m] = ar.shape.slice(-2);
		const diagSize = Math.min(n, m - offset);
		ar = ar.reshape([...ar.shape.slice(0, -2), n * m]);
		toDispose.push(ar);
		const npad = diagSize * (m + 1) - n * m;
		if (npad > 0) {
			ar = pad$1(ar, [...require_backend.rep(ar.ndim - 1, [0, 0]), [0, npad]]);
			toDispose.push(ar);
		} else if (npad < 0) {
			ar = shrink(ar, [...ar.shape.slice(0, -1), n * m + npad].map((x) => [0, x]));
			toDispose.push(ar);
		}
		ar = ar.reshape([
			...ar.shape.slice(0, -1),
			diagSize,
			m + 1
		]);
		toDispose.push(ar);
		const sliced = shrink(ar, [...ar.shape.slice(0, -1).map((x) => [0, x]), [offset, offset + 1]]);
		toDispose.push(sliced);
		const result = sliced.reshape(ar.shape.slice(0, -1));
		for (const v of toDispose) if (v !== result) v[Symbol.dispose]();
		return result;
	}
	/** Flatten the array without changing its data. */
	flatten() {
		return this.reshape(-1);
	}
	/** Flatten the array without changing its data. */
	ravel() {
		return this.reshape(-1);
	}
	/**
	* Iterate over the first dimension of this array, returning slices.
	*
	* This can be used to destructure arrays. For example:
	*
	* ```js
	* let x = np.array([[1, 2], [3, 4]]);
	* let [a, b] = x;
	* console.log(a.js()); // [1, 2]
	* console.log(b.js()); // [3, 4]
	* ```
	*/
	*[Symbol.iterator]() {
		if (this.ndim === 0) throw new Error("Cannot iterate over a scalar array");
		let residual = this;
		const n = this.shape[0];
		const subarrayShape = this.shape.slice(1);
		for (let i = 0; i < n; i++) if (i < n - 1) {
			const lr = split$2(residual, 0, [1, residual.shape[0] - 1]);
			if (residual !== this) residual.dispose();
			const first = lr[0];
			const reshaped = first.reshape(subarrayShape);
			if (first !== reshaped) first.dispose();
			yield reshaped;
			residual = lr[1];
		} else {
			const reshaped = residual.reshape(subarrayShape);
			if (residual !== reshaped && residual !== this) residual.dispose();
			yield reshaped;
		}
	}
	/**
	* Return a sorted copy of an array in ascending order.
	*
	* See `jax.numpy.sort` for full docs.
	*/
	sort(axis = -1) {
		axis = require_backend.checkAxis(axis, this.ndim);
		if (this.shape[axis] <= 1) return this.reshape(this.shape);
		if (axis === this.ndim - 1) return sort$1(this);
		const perm = require_backend.range(this.ndim);
		perm.splice(axis, 1);
		perm.push(axis);
		const transposed = this.transpose(perm);
		const sorted = sort$1(transposed);
		const result = sorted.transpose(require_backend.invertPermutation(perm));
		if (transposed !== this) transposed.dispose();
		if (sorted !== result) sorted.dispose();
		return result;
	}
	/**
	* Return the indices that would sort an array. Unlike `sort`, this is
	* guaranteed to be a stable sorting algorithm; it always returns the smaller
	* index first in event of ties.
	*
	* See `jax.numpy.argsort` for full docs.
	*/
	argsort(axis = -1) {
		axis = require_backend.checkAxis(axis, this.ndim);
		if (axis === this.ndim - 1) {
			const [y$1, yi$1] = argsort$1(this);
			y$1.dispose();
			return yi$1;
		}
		const perm = require_backend.range(this.ndim);
		perm.splice(axis, 1);
		perm.push(axis);
		const transposed = this.transpose(perm);
		const [y, yi] = argsort$1(transposed);
		y.dispose();
		if (transposed !== this) transposed.dispose();
		const result = yi.transpose(require_backend.invertPermutation(perm));
		if (yi !== result) yi.dispose();
		return result;
	}
	/**
	* Slice an array along one or more axes.
	*
	* This is the equivalent of slicing in Python, e.g. `x[1:3, 2, :, None]`. To
	* mimic this in JavaScript, we would write:
	*
	* ```js
	* x.slice([1, 3], 2, [], null);
	* ```
	*
	* The `slice` method accepts a variable number of arguments, each of which
	* can be a number, an empty array, a single-element array, a two-element
	* array, or `null`. The arguments are interpreted as follows:
	*
	* - A number `n` means to access the `n`-th element along that axis, removing
	*   that axis from the resulting shape.
	* - An empty array `[]` means to keep that axis as-is, like `:` in Python.
	* - A single-element array `[i]` means to start slicing from index `i`
	*   (inclusive) to the end of the axis, like `x[i:]`.
	* - A two-element array `[i, j]` means to slice from index `i` (inclusive)
	*   to index `j` (exclusive), like `x[i:j]`.
	* - `null` means to add a new axis at that position, like `np.newaxis`.
	*
	* Like in Python, negative indices are supported, which count from the end of
	* the axis. For example, `-1` means the last element.
	*
	* Strided slices are not yet implemented, so you cannot write `x[::2]` or
	* similar.
	*
	* Advanced indexing by integer arrays is also supported. This translates to
	* the "gather" primitive, and it allows you to access specific elements of
	* the array by integer indices stored in another array.
	*/
	slice(...index) {
		const checkBounds = (n, i) => {
			if (i > n || i < -n) throw new RangeError(`Index ${i} out of bounds for axis of size ${n}`);
			return i < 0 ? n + i : i;
		};
		const hasAdvancedIdx = index.some((value) => value instanceof Tracer);
		const axesForGather = [];
		let outDim = -1;
		if (hasAdvancedIdx) {
			const advancedAxes = [];
			let currentAxisForGather = 0;
			for (let i = 0; i < index.length; i++) {
				const value = index[i];
				if (value instanceof Tracer) {
					advancedAxes.push(i);
					axesForGather.push(currentAxisForGather++);
				} else if (typeof value === "number") advancedAxes.push(i);
				else currentAxisForGather++;
			}
			if (advancedAxes[advancedAxes.length - 1] - advancedAxes[0] !== advancedAxes.length - 1) outDim = 0;
			else outDim = axesForGather[0];
		}
		const slice = [];
		const basicShape = [];
		let needsReshape = false;
		let axis = 0;
		for (const value of index) if (value === null) {
			basicShape.push(1);
			needsReshape = true;
		} else if (typeof value === "number") {
			if (axis >= this.shape.length) throw new RangeError("Too many indices");
			const i = checkBounds(this.shape[axis++], value);
			slice.push([i, i + 1]);
			needsReshape = true;
		} else if (Array.isArray(value)) {
			if (axis >= this.shape.length) throw new RangeError("Too many indices");
			const n = this.shape[axis++];
			if (value.length === 0) {
				basicShape.push(n);
				slice.push([0, n]);
			} else if (value.length === 1) {
				const i = checkBounds(n, value[0]);
				basicShape.push(n - i);
				slice.push([i, n]);
			} else if (value.length === 2) {
				const [i, j] = value.map((v) => checkBounds(n, v));
				if (i > j) throw new RangeError(`Slice start at ${i} > end at ${j}`);
				basicShape.push(j - i);
				slice.push([i, j]);
			}
		} else if (value instanceof Tracer) {
			const n = this.shape[axis++];
			basicShape.push(n);
			slice.push([0, n]);
		} else throw new TypeError(`Invalid slice argument: ${JSON.stringify(value)}`);
		while (axis < this.shape.length) {
			slice.push([0, this.shape[axis]]);
			basicShape.push(this.shape[axis++]);
		}
		let result = shrink(this, slice);
		if (needsReshape) {
			const prev = result;
			result = reshape$1(result, basicShape);
			if (prev !== result) prev[Symbol.dispose]?.();
		}
		if (hasAdvancedIdx) {
			const prev = result;
			result = gather(result, index.filter((a) => a instanceof Tracer), axesForGather, outDim);
			if (prev !== result) prev[Symbol.dispose]?.();
		}
		return result;
	}
};
function ndim$1(x) {
	if (x instanceof Tracer) return x.shape.length;
	else return 0;
}
function getShape(x) {
	return x instanceof Tracer ? x.shape : [];
}
var ShapedArray = class ShapedArray {
	constructor(shape$1, dtype, weakType) {
		this.shape = shape$1;
		this.dtype = dtype;
		this.weakType = weakType;
	}
	static fromAval(aval) {
		return new ShapedArray(aval.shape, aval.dtype, aval.weakType);
	}
	get ndim() {
		return this.shape.length;
	}
	get size() {
		return require_backend.prod(this.shape);
	}
	scalar() {
		return new ShapedArray([], this.dtype, this.weakType);
	}
	toString() {
		return `${this.dtype}[${this.shape.join(",")}]`;
	}
	equals(other) {
		return this === other || this.constructor === other.constructor && this.ndim === other.ndim && this.shape.every((d, i) => d === other.shape[i]);
	}
};
function getAval(x) {
	if (x instanceof Tracer) return x.aval;
	else if (typeof x === "boolean" || typeof x === "number") return new ShapedArray([], typeof x === "boolean" ? require_backend.DType.Bool : require_backend.DType.Float32, typeof x === "boolean" ? false : true);
	else throw new TypeError(`Unknown value: ${x}`);
}
/**
* When non-null, bind() pushes EvalTrace-level results (level 0) into this
* array. Used by partialEvalFlat to track intermediate concrete Arrays created
* during PE scope — these bypass PE tracking and would otherwise leak.
* Activated in the Array constructor when set (not null).
*/
let _peArrayCreationTracker = null;
function _setPACT(v) {
	_peArrayCreationTracker = v;
}
function bind(prim, args, params = {}) {
	const topTrace = findTopTrace(args);
	const tracers = args.map((arg) => fullRaise(topTrace, arg));
	const outs = topTrace.processPrimitive(prim, tracers, params);
	if (require_backend.DEBUG >= 5) console.info(`processing rule for ${prim} on ${tracers.map((x) => x.toString())} and got ${outs.map((x) => x.toString())}`);
	const lowered = outs.map((out) => out.fullLower());
	for (let i = 0; i < args.length; i++) if (!(args[i] instanceof Tracer)) tracers[i][Symbol.dispose]();
	return lowered;
}
function findTopTrace(xs) {
	let topMain = traceStack[0];
	for (const x of xs) if (x instanceof Tracer && x._trace.main.level > topMain.level) topMain = x._trace.main;
	if (dynamicTrace && dynamicTrace.level > topMain.level) topMain = dynamicTrace;
	return new topMain.traceType(topMain);
}
function fullRaise(trace$1, val) {
	if (!(val instanceof Tracer)) return trace$1.pure(val);
	const level = trace$1.main.level;
	if (Object.is(val._trace.main, trace$1.main)) return val;
	else if (val._trace.main.level < level) return trace$1.lift(val);
	else if (val._trace.main.level > level) throw new Error(`Can't lift Tracer level ${val._trace.main.level} to level ${level}`);
	else throw new Error(`Different traces at same level: ${val._trace.constructor}, ${trace$1.constructor}.`);
}
var TreeMismatchError = class extends TypeError {
	constructor(where$2, left, right) {
		super(`Mismatched tree structures in ${where$2}: ${left} != ${right}`);
	}
};
/** Flatten a function of `JsTree` input/output for use in tracing. */
function flattenFun(f, inTree) {
	const store = { value: void 0 };
	const flatFun = (...argsFlat) => {
		const pytreeArgs = unflatten(inTree, argsFlat);
		const out = f(...pytreeArgs);
		const [outFlat, outTree] = flatten(out);
		store.value = outTree;
		return outFlat;
	};
	return [flatFun, store];
}
/** Like flattenFun, but expects f to return [main, aux] tuple. */
function flattenFunWithAux(f, inTree) {
	const store = { value: void 0 };
	const auxStore = { value: void 0 };
	const flatFun = (...argsFlat) => {
		const pytreeArgs = unflatten(inTree, argsFlat);
		const result = f(...pytreeArgs);
		if (!Array.isArray(result) || result.length !== 2) throw new Error("Function with `hasAux: true` must return [output, aux] tuple");
		const [out, aux] = result;
		const [outFlat, outTree] = flatten(out);
		store.value = outTree;
		auxStore.value = aux;
		return outFlat;
	};
	return [
		flatFun,
		store,
		auxStore
	];
}
var UseAfterFreeError = class extends ReferenceError {
	constructor(tracer) {
		super(`Referenced tracer ${tracer.toString()} has been disposed`);
	}
};

//#endregion
//#region src/frontend/jaxpr.ts
var import_usingCtx$11 = /* @__PURE__ */ __toESM(require_usingCtx(), 1);
/** Variable in a Jaxpr expression. */
var Var = class Var {
	static #nextId = 1;
	id;
	aval;
	constructor(aval) {
		this.id = Var.#nextId++;
		this.aval = aval;
	}
	toString() {
		return `Var(${this.id}):${this.aval.toString()}`;
	}
};
/** Literal in a Jaxpr expression. Currently, only scalars are supported. */
var Lit = class {
	value;
	aval;
	get dtype() {
		return this.aval.dtype;
	}
	constructor(aval, value) {
		if (aval.shape.length !== 0) throw new Error(`internal: Lit must be a scalar`);
		this.value = value;
		this.aval = ShapedArray.fromAval(aval);
	}
};
function atomIsLit(atom, literal) {
	return atom instanceof Lit && (literal === void 0 || atom.value === literal);
}
var VarPrinter = class {
	names = /* @__PURE__ */ new Map();
	#next = "a";
	#advance() {
		const ret = this.#next;
		let lastNonz = this.#next.length - 1;
		while (lastNonz >= 0 && this.#next[lastNonz] === "z") lastNonz--;
		if (lastNonz < 0) this.#next = "a".repeat(this.#next.length + 1);
		else {
			let result = this.#next.slice(0, lastNonz);
			result += String.fromCharCode(this.#next.charCodeAt(lastNonz) + 1);
			result += "a".repeat(this.#next.length - 1 - lastNonz);
			this.#next = result;
		}
		return ret;
	}
	name(v) {
		if (this.names.has(v)) return this.names.get(v);
		const name = this.#advance();
		this.names.set(v, name);
		return name;
	}
	nameType(v) {
		return `${this.name(v)}:${v.aval.toString()}`;
	}
};
/** A single statement / binding in a Jaxpr, in ANF form. */
var JaxprEqn = class {
	constructor(primitive, inputs, params, outBinders) {
		this.primitive = primitive;
		this.inputs = inputs;
		this.params = params;
		this.outBinders = outBinders;
	}
	pprint(usedVars, vp = new VarPrinter()) {
		const lhs = require_backend.PPrint.pp(this.outBinders.map((v) => !usedVars || usedVars.has(v) ? vp.nameType(v) : "_").join(" "));
		let rhs = require_backend.PPrint.pp(this.primitive);
		const paramsList = Object.entries(this.params).map(([k, v]) => require_backend.PPrint.pp(`${k}=${v}`));
		if (paramsList.length > 0) rhs = rhs.stack(require_backend.PPrint.pp(" [ ")).stack(require_backend.PPrint.prototype.concat(...paramsList)).stack(require_backend.PPrint.pp(" ] "));
		else rhs = rhs.stack(require_backend.PPrint.pp(" "));
		rhs = rhs.stack(require_backend.PPrint.pp(this.inputs.map((x) => x instanceof Var ? vp.name(x) : String(x.value)).join(" ")));
		return lhs.stack(require_backend.PPrint.pp(" = ")).stack(rhs);
	}
	toString() {
		return this.pprint().toString();
	}
};
/** Typed intermediate representation for traced computations. */
var Jaxpr = class Jaxpr {
	#hash;
	constructor(inBinders, eqns, outs) {
		this.inBinders = inBinders;
		this.eqns = eqns;
		this.outs = outs;
	}
	pprint() {
		const vp = new VarPrinter();
		const usedVars = new Set([...this.outs, ...this.eqns.flatMap((eqn) => eqn.inputs)].filter((x) => x instanceof Var));
		const inBinders = this.inBinders.map((v) => vp.nameType(v)).join(", ");
		const eqns = require_backend.PPrint.prototype.concat(...this.eqns.map((e$1) => e$1.pprint(usedVars, vp)));
		const outs = this.outs.map((x) => x instanceof Var ? vp.name(x) : x.value).join(", ");
		return require_backend.PPrint.pp(`{ lambda ${inBinders} .`).concat((this.eqns.length ? require_backend.PPrint.pp("let ").stack(eqns).concat(require_backend.PPrint.pp(`in ( ${outs} ) }`)) : require_backend.PPrint.pp(`( ${outs} ) }`)).indent(2));
	}
	toString() {
		return this.pprint().toString();
	}
	/**
	* Gets a hash of this Jaxpr.
	*
	* Var identity is not considered in the hash, so two Jaxprs with the same
	* order of assignments and operators but different variable IDs will resolve
	* to the same hash (and toString representation).
	*/
	getHash() {
		if (this.#hash !== void 0) return this.#hash;
		const hasher = new require_backend.FpHash();
		const varIds = /* @__PURE__ */ new Map();
		const vi = (v) => {
			if (varIds.has(v)) return varIds.get(v);
			const id = varIds.size + 1;
			varIds.set(v, require_backend.FpHash.hash(id, v.aval.dtype, ...v.aval.shape));
			return id;
		};
		hasher.update(this.inBinders.length);
		for (const x of this.inBinders) hasher.update(vi(x));
		hasher.update(this.eqns.length);
		for (const eqn of this.eqns) {
			hasher.update(eqn.primitive);
			hasher.update(eqn.inputs.length);
			for (const x of eqn.inputs) hasher.update(x instanceof Var ? vi(x) : x.value);
			hasher.update(JSON.stringify(eqn.params));
			hasher.update(eqn.outBinders.length);
			for (const x of eqn.outBinders) hasher.update(vi(x));
		}
		hasher.update(this.outs.length);
		for (const x of this.outs) hasher.update(x instanceof Var ? vi(x) : x.value);
		return this.#hash = hasher.value;
	}
	hash(state) {
		state.update(this.getHash());
	}
	/**
	* Produce a simplified Jaxpr with basic optimizations applied.
	*  - Trim away unused variables.
	*  - Fold away *1, *0, or +0 operations against literals.
	*  - Remove no-op movement operations.
	*/
	simplify() {
		const context = /* @__PURE__ */ new Map();
		const newEqns = [];
		for (const e$1 of this.eqns) {
			const inputs = e$1.inputs.map((x) => x instanceof Var ? context.get(x) ?? x : x);
			const eqn = new JaxprEqn(e$1.primitive, inputs, e$1.params, e$1.outBinders);
			if (eqn.primitive === Primitive.Add) {
				const [a, b] = inputs;
				const c = eqn.outBinders[0];
				if (atomIsLit(a, 0)) context.set(c, b);
				else if (atomIsLit(b, 0)) context.set(c, a);
				else if (atomIsLit(a) && atomIsLit(b)) context.set(c, new Lit(promoteAvals(a.aval, b.aval), a.dtype === require_backend.DType.Bool ? Math.min(a.value + b.value, 1) : a.value + b.value));
				else newEqns.push(eqn);
			} else if (eqn.primitive === Primitive.Neg) {
				const [a] = inputs;
				const c = eqn.outBinders[0];
				if (atomIsLit(a)) context.set(c, new Lit(a.aval, -a.value));
				else newEqns.push(eqn);
			} else if (eqn.primitive === Primitive.Mul) {
				const [a, b] = inputs;
				const c = eqn.outBinders[0];
				if (atomIsLit(a, 1)) context.set(c, b);
				else if (atomIsLit(b, 1)) context.set(c, a);
				else if (atomIsLit(a) && atomIsLit(b)) context.set(c, new Lit(promoteAvals(a.aval, b.aval), a.value * b.value));
				else newEqns.push(eqn);
			} else if (eqn.primitive === Primitive.Idiv) {
				const [a, b] = inputs;
				const c = eqn.outBinders[0];
				if (atomIsLit(b, 1) && !require_backend.isFloatDtype(a.aval.dtype)) context.set(c, a);
				else newEqns.push(eqn);
			} else if ((eqn.primitive === Primitive.Broadcast || eqn.primitive === Primitive.Reshape) && require_backend.deepEqual(eqn.params.shape, eqn.inputs[0].aval.shape) || eqn.primitive === Primitive.Transpose && eqn.params.perm.every((p, i) => p === i) || eqn.primitive === Primitive.Flip && eqn.params.axis.length === 0 || eqn.primitive === Primitive.Shrink && eqn.params.slice.every(([s, e$2], i) => s === 0 && e$2 === eqn.inputs[0].aval.shape[i]) || eqn.primitive === Primitive.Pad && eqn.params.width.every(([w0, w1]) => w0 === 0 && w1 === 0)) context.set(eqn.outBinders[0], eqn.inputs[0]);
			else newEqns.push(eqn);
		}
		const outs = this.outs.map((x) => x instanceof Var ? context.get(x) ?? x : x);
		const usedVars = new Set(outs.filter((x) => x instanceof Var));
		const liveEqns = [];
		for (let i = newEqns.length - 1; i >= 0; i--) {
			const eqn = newEqns[i];
			if (eqn.outBinders.some((v) => usedVars.has(v))) {
				liveEqns.push(eqn);
				for (const v of eqn.inputs) if (v instanceof Var) usedVars.add(v);
			}
		}
		return new Jaxpr(this.inBinders, liveEqns.reverse(), outs);
	}
	/** Flattens nested Jit in a Jaxpr. Useful for handling jit-of-jit. */
	flatten() {
		if (!this.eqns.some((eqn) => eqn.primitive === Primitive.Jit)) return this;
		const newEqns = [];
		const varMap = /* @__PURE__ */ new Map();
		const varMapF = (x) => x instanceof Var ? varMap.get(x) ?? x : x;
		for (const eqn of this.eqns) if (eqn.primitive === Primitive.Jit) {
			const jaxpr = eqn.params.jaxpr.flatten();
			const translation = /* @__PURE__ */ new Map();
			const translationF = (x) => x instanceof Var ? translation.get(x) : x;
			for (const [v, x] of require_backend.zip(jaxpr.inBinders, eqn.inputs)) translation.set(v, varMapF(x));
			for (const ieqn of jaxpr.eqns) {
				const inputs = ieqn.inputs.map(translationF);
				const outBinders = [];
				for (const v of ieqn.outBinders) {
					const u = new Var(v.aval);
					outBinders.push(u);
					translation.set(v, u);
				}
				newEqns.push(new JaxprEqn(ieqn.primitive, inputs, ieqn.params, outBinders));
			}
			for (const [v, x] of require_backend.zip(eqn.outBinders, jaxpr.outs)) varMap.set(v, translationF(x));
		} else if (eqn.inputs.some((x) => x instanceof Var && varMap.has(x))) newEqns.push(new JaxprEqn(eqn.primitive, eqn.inputs.map(varMapF), eqn.params, eqn.outBinders));
		else newEqns.push(eqn);
		const newOuts = this.outs.map(varMapF);
		return new Jaxpr(this.inBinders, newEqns, newOuts);
	}
};
var JaxprType = class {
	constructor(inTypes, outTypes) {
		this.inTypes = inTypes;
		this.outTypes = outTypes;
	}
	toString() {
		const inTypes = this.inTypes.map((aval) => aval.toString()).join(", ");
		const outTypes = this.outTypes.map((aval) => aval.toString()).join(", ");
		return `(${inTypes}) -> (${outTypes})`;
	}
};
function typecheckJaxpr(jaxpr) {
	const env = /* @__PURE__ */ new Set();
	for (const v of jaxpr.inBinders) {
		if (env.has(v)) throw new TypeError(`Duplicate variable binding: ${v}`);
		env.add(v);
	}
	for (const eqn of jaxpr.eqns) {
		const inTypes$1 = eqn.inputs.map((x) => typecheckAtom(env, x));
		const rule = abstractEvalRules[eqn.primitive];
		const outTypes$1 = rule(inTypes$1, eqn.params);
		for (const [outBinder, outType] of require_backend.zip(eqn.outBinders, outTypes$1)) {
			if (!outType.equals(outBinder.aval)) throw new TypeError(`Output binder type mismatch in ${eqn.primitive}: ${outBinder} vs ${outType}`);
			if (env.has(outBinder)) throw new TypeError(`Duplicate variable binding: ${outBinder}`);
			env.add(outBinder);
		}
	}
	const inTypes = jaxpr.inBinders.map((v) => v.aval);
	const outTypes = jaxpr.outs.map((x) => typecheckAtom(env, x));
	return new JaxprType(inTypes, outTypes);
}
function typecheckAtom(env, x) {
	if (x instanceof Var) {
		if (!env.has(x)) throw new Error(`Unknown variable: ${x}`);
		return x.aval;
	} else if (x instanceof Lit) return x.aval;
	else throw new TypeError(`Invalid atom type: ${x}`);
}
/** Evaluate a Jaxpr on an array of inputs. */
function evalJaxpr(jaxpr, args) {
	const env = /* @__PURE__ */ new Map();
	const usageCount = /* @__PURE__ */ new Map();
	for (const x of jaxpr.eqns.flatMap((eqn) => eqn.inputs).concat(jaxpr.outs)) if (x instanceof Var) usageCount.set(x, (usageCount.get(x) ?? 0) + 1);
	const outputVars = /* @__PURE__ */ new Set();
	for (const x of jaxpr.outs) if (x instanceof Var) outputVars.add(x);
	const remainingRefs = /* @__PURE__ */ new Map();
	const read = (x) => {
		if (x instanceof Var) return env.get(x);
		else {
			const arr = array(x.value, { dtype: x.dtype });
			if (arr instanceof Array$1) anonymousConstArrays.add(arr);
			return arr;
		}
	};
	const write = (v, val) => {
		if (env.has(v)) throw new Error(`Variable already bound: ${v}`);
		const refCount = usageCount.get(v) ?? 0;
		if (refCount) {
			env.set(v, val);
			remainingRefs.set(v, refCount);
		} else val.dispose();
	};
	const consumeRead = (x) => {
		if (x instanceof Var) {
			const left = remainingRefs.get(x);
			remainingRefs.set(x, left - 1);
			if (left === 1 && !outputVars.has(x)) {
				env.get(x)?.dispose();
				env.delete(x);
			}
		}
	};
	try {
		for (const [v, arg] of require_backend.zip(jaxpr.inBinders, args)) write(v, arg);
		for (const eqn of jaxpr.eqns) {
			const inVals = eqn.inputs.map(read);
			const outVals = bind(eqn.primitive, inVals, eqn.params);
			for (const [v, val] of require_backend.zip(eqn.outBinders, outVals)) write(v, val);
			for (const x of eqn.inputs) consumeRead(x);
		}
		return jaxpr.outs.map(read);
	} catch (error) {
		for (const val of env.values()) val.dispose();
		throw error;
	}
}
/** Convert a Jaxpr to a callable function by evaluating it. */
function jaxprAsFun(jaxpr) {
	return (...args) => evalJaxpr(jaxpr, args);
}
/** Jaxpr with a collection of associated, traced constants. */
var ClosedJaxpr = class ClosedJaxpr {
	constructor(jaxpr, consts) {
		this.jaxpr = jaxpr;
		this.consts = consts;
	}
	/** String representation of this Jaxpr. */
	toString() {
		return this.jaxpr.toString();
	}
	/** Apply a function to the underlying Jaxpr. */
	mapJaxpr(f) {
		return new ClosedJaxpr(f(this.jaxpr), this.consts);
	}
	/** Dispose of the constants in this Jaxpr. */
	dispose() {
		for (const c of this.consts) c.dispose();
	}
};
/** Tracer that records its operations to dynamically construct a Jaxpr. */
var JaxprTracer = class extends Tracer {
	#rc;
	constructor(trace$1, aval) {
		super(trace$1);
		this.aval = aval;
		this.#rc = 1;
	}
	toString() {
		return `JaxprTracer(${this.aval.toString()})`;
	}
	get ref() {
		this.#rc++;
		return this;
	}
	dispose() {
		this.#rc--;
	}
	/** Number of live references the user holds (1 + .ref count − .dispose count). */
	get refCount() {
		return this.#rc;
	}
	trackLiftedConstant() {
		this.#rc++;
	}
};
/** Analogous to the 'DynamicJaxprTrace' class in JAX. */
var JaxprTrace = class extends Trace {
	/** Register a Jaxpr argument with a given shape and return the tracer. */
	newArg(aval) {
		aval = ShapedArray.fromAval(aval);
		const tracer = this.builder.newTracer(this, aval);
		this.builder.addVar(tracer);
		return tracer;
	}
	/** Register a constant / literal in this Jaxpr. */
	getOrMakeConstTracer(val) {
		let ownedByBuilder = false;
		if (!(val instanceof Tracer)) {
			val = pureArray(val);
			ownedByBuilder = true;
		} else if (val instanceof Array$1 && anonymousConstArrays.has(val)) {
			ownedByBuilder = true;
			anonymousConstArrays.delete(val);
		}
		let tracer = this.builder.constTracers.get(val);
		if (tracer === void 0) {
			tracer = this.builder.newTracer(this, ShapedArray.fromAval(getAval(val)));
			if (!ownedByBuilder) val.ref;
			this.builder.addConst(tracer, val);
		} else tracer.trackLiftedConstant();
		return tracer;
	}
	pure = this.getOrMakeConstTracer;
	lift = this.getOrMakeConstTracer;
	processPrimitive(primitive, tracers, params) {
		const avalsIn = tracers.map((t) => t.aval);
		const avalsOut = abstractEvalRules[primitive](avalsIn, params);
		const outTracers = avalsOut.map((aval) => this.builder.newTracer(this, aval));
		const eqn = new JaxprEqn(primitive, tracers.map((t) => this.builder.getVar(t)), params, outTracers.map((t) => this.builder.addVar(t)));
		this.builder.addEqn(eqn);
		return outTracers;
	}
	get builder() {
		return this.main.globalData;
	}
};
/** Incrementally constructs a Jaxpr. */
var JaxprBuilder = class {
	eqns = [];
	tracerToVar = /* @__PURE__ */ new Map();
	constTracers = /* @__PURE__ */ new Map();
	constVals = /* @__PURE__ */ new Map();
	tracers = [];
	newTracer(trace$1, aval) {
		const tracer = new JaxprTracer(trace$1, aval);
		this.tracers.push(tracer);
		return tracer;
	}
	addEqn(eqn) {
		this.eqns.push(eqn);
	}
	addVar(tracer) {
		if (this.tracerToVar.has(tracer)) throw new Error(`Tracer was added as variable twice: ${tracer}`);
		const v = new Var(tracer.aval);
		this.tracerToVar.set(tracer, v);
		return v;
	}
	getVar(tracer) {
		const v = this.tracerToVar.get(tracer);
		if (v === void 0) throw new Error(`Could not find variable for tracer: ${tracer}`);
		return v;
	}
	addConst(tracer, val) {
		const v = this.addVar(tracer);
		this.constTracers.set(val, tracer);
		this.constVals.set(v, val);
		return v;
	}
	build(inTracers, outTracers) {
		const [constVars, consts] = require_backend.unzip2(this.constVals.entries());
		const t2v = this.getVar.bind(this);
		const inBinders = [...constVars, ...inTracers.map(t2v)];
		const outVars = outTracers.map(t2v);
		const jaxpr = new Jaxpr(inBinders, this.eqns, outVars);
		typecheckJaxpr(jaxpr);
		const cjaxpr = new ClosedJaxpr(jaxpr, consts);
		return _inlineLiterals(cjaxpr);
	}
};
function _inlineLiterals({ jaxpr, consts }) {
	const literals = /* @__PURE__ */ new Map();
	const constBinders = [];
	const newConsts = [];
	for (let i = 0; i < consts.length; i++) if (ndim$1(consts[i]) === 0 && consts[i] instanceof Array$1) {
		const ar = consts[i];
		literals.set(jaxpr.inBinders[i], new Lit(ar.aval, ar.dataSync()[0]));
		ar.dispose();
	} else {
		constBinders.push(jaxpr.inBinders[i]);
		newConsts.push(consts[i]);
	}
	const newEqns = jaxpr.eqns.map((eqn) => new JaxprEqn(eqn.primitive, eqn.inputs.map((x) => literals.get(x) ?? x), eqn.params, eqn.outBinders));
	const newOuts = jaxpr.outs.map((x) => literals.get(x) ?? x);
	const newJaxpr = new Jaxpr([...constBinders, ...jaxpr.inBinders.slice(consts.length)], newEqns, newOuts);
	typecheckJaxpr(newJaxpr);
	return new ClosedJaxpr(newJaxpr, newConsts);
}
function binopAbstractEval([x, y]) {
	if (!(x instanceof ShapedArray) || !(y instanceof ShapedArray)) throw new TypeError("binopAbstractEval expects ShapedArray inputs");
	return [promoteAvals(x, y)];
}
function compareAbstractEval([x, y]) {
	if (!(x instanceof ShapedArray) || !(y instanceof ShapedArray)) throw new TypeError("compareAbstractEval expects ShapedArray inputs");
	const aval = promoteAvals(x, y);
	return [new ShapedArray(aval.shape, require_backend.DType.Bool, false)];
}
function vectorizedUnopAbstractEval([x]) {
	return [ShapedArray.fromAval(x)];
}
const abstractEvalRules = {
	[Primitive.Add]: binopAbstractEval,
	[Primitive.Mul]: binopAbstractEval,
	[Primitive.Idiv]: binopAbstractEval,
	[Primitive.Mod]: binopAbstractEval,
	[Primitive.Min]: binopAbstractEval,
	[Primitive.Max]: binopAbstractEval,
	[Primitive.Neg]: vectorizedUnopAbstractEval,
	[Primitive.Reciprocal]: vectorizedUnopAbstractEval,
	[Primitive.Floor]: vectorizedUnopAbstractEval,
	[Primitive.Ceil]: vectorizedUnopAbstractEval,
	[Primitive.StopGradient]: vectorizedUnopAbstractEval,
	[Primitive.Cast]([x], { dtype }) {
		return [new ShapedArray(x.shape, dtype, false)];
	},
	[Primitive.Bitcast]([x], { dtype }) {
		if (x.dtype === require_backend.DType.Bool || dtype === require_backend.DType.Bool) throw new TypeError("Bitcast to/from bool is not allowed");
		if (require_backend.byteWidth(x.dtype) !== require_backend.byteWidth(dtype)) throw new TypeError(`Bitcast from ${x.dtype} to ${dtype} with different byte width`);
		return [new ShapedArray(x.shape, dtype, false)];
	},
	[Primitive.Sin]: vectorizedUnopAbstractEval,
	[Primitive.Cos]: vectorizedUnopAbstractEval,
	[Primitive.Asin]: vectorizedUnopAbstractEval,
	[Primitive.Atan]: vectorizedUnopAbstractEval,
	[Primitive.Exp]: vectorizedUnopAbstractEval,
	[Primitive.Log]: vectorizedUnopAbstractEval,
	[Primitive.Erf]: vectorizedUnopAbstractEval,
	[Primitive.Erfc]: vectorizedUnopAbstractEval,
	[Primitive.Sqrt]: vectorizedUnopAbstractEval,
	[Primitive.Reduce]([x], { axis }) {
		const axisSet = new Set(axis);
		const newShape = x.shape.filter((_, i) => !axisSet.has(i));
		return [new ShapedArray(newShape, x.dtype, x.weakType)];
	},
	[Primitive.Pool]([x], { window, strides }) {
		const shape$1 = checkPoolShape(x.shape, window, strides);
		return [new ShapedArray(shape$1, x.dtype, x.weakType)];
	},
	[Primitive.PoolTranspose]([x], { inShape, window, strides }) {
		const shape$1 = checkPoolShape(inShape, window, strides);
		if (!require_backend.deepEqual(shape$1, x.shape)) throw new TypeError(`PoolTranspose shape mismatch: expected ${JSON.stringify(shape$1)}, got ${JSON.stringify(x.shape)}`);
		return [new ShapedArray(inShape, x.dtype, x.weakType)];
	},
	[Primitive.Dot]([x, y]) {
		if (x.ndim === 0 && y.ndim === 0) throw new TypeError("Dot requires at least 1D inputs");
		const { shape: shape$1, dtype, weakType } = promoteAvals(x, y);
		shape$1.splice(-1, 1);
		return [new ShapedArray(shape$1, dtype, weakType)];
	},
	[Primitive.Conv]([lhs, rhs], params) {
		const { dtype, weakType } = promoteAvals(lhs.scalar(), rhs.scalar());
		const shape$1 = checkConvShape(lhs.shape, rhs.shape, params);
		return [new ShapedArray(shape$1, dtype, weakType)];
	},
	[Primitive.Compare]: compareAbstractEval,
	[Primitive.Where]([cond, x, y]) {
		if (cond.dtype !== require_backend.DType.Bool) throw new TypeError(`Condition must be boolean, got ${cond.dtype}`);
		const xy = promoteAvals(x, y);
		const shape$1 = require_backend.generalBroadcast(cond.shape, xy.shape);
		return [new ShapedArray(shape$1, xy.dtype, xy.weakType)];
	},
	[Primitive.Concatenate](xs, { axis }) {
		if (xs.length === 0) throw new TypeError("Concatenate requires at least one input");
		for (const x of xs) if (x.ndim !== xs[0].ndim || !x.shape.every((s, i) => i === axis || s === xs[0].shape[i])) throw new TypeError(`Concatenate: inputs ${xs[0]} and ${x} must match shapes except on axis ${axis}`);
		const shape$1 = xs[0].shape.slice();
		shape$1[axis] = xs.reduce((sum$1, x) => sum$1 + x.shape[axis], 0);
		const { dtype, weakType } = xs.map((x) => x.scalar()).reduce(promoteAvals);
		return [new ShapedArray(shape$1, dtype, weakType)];
	},
	[Primitive.Split]([x], { axis, sizes }) {
		const totalSize = sizes.reduce((a, b) => a + b, 0);
		if (x.shape[axis] !== totalSize) throw new TypeError(`Split: sizes ${sizes} do not sum to dimension ${x.shape[axis]} on axis ${axis}`);
		return sizes.map((size$1) => {
			return new ShapedArray(x.shape.toSpliced(axis, 1, size$1), x.dtype, x.weakType);
		});
	},
	[Primitive.RandomBits]([k0, k1], { shape: shape$1 }) {
		if (k0.dtype !== require_backend.DType.Uint32 || k1.dtype !== require_backend.DType.Uint32) throw new TypeError(`RandomBits requires uint32 keys, got ${k0.dtype} and ${k1.dtype}`);
		if (!require_backend.deepEqual(k0.shape, k1.shape)) throw new TypeError(`RandomBits: Keys have different shapes ${k0.shape} and ${k1.shape}`);
		if (!require_backend.deepEqual(shape$1.slice(0, k0.ndim), k0.shape)) throw new TypeError(`RandomBits: generated shape ${shape$1} must match key shape ${k0.shape}`);
		return [new ShapedArray(shape$1, require_backend.DType.Uint32, false)];
	},
	[Primitive.Gather]([x, ...indices], { axis, outDim }) {
		for (const a of indices) if (a.dtype !== require_backend.DType.Int32 && a.dtype !== require_backend.DType.Uint32) throw new TypeError(`Gather indices must be Int32 or Uint32, got ${a.dtype}`);
		if (axis.length !== indices.length) throw new TypeError(`Gather: ${axis} axes but ${indices.length} indices`);
		if (indices.length === 0) throw new TypeError("Gather must have 1+ indices with same shape");
		if (axis.some((a) => a < 0 || a >= x.shape.length)) throw new TypeError("Gather axis out of bounds");
		if (outDim < 0 || outDim > x.shape.length - axis.length) throw new TypeError("Gather outDim out of bounds");
		const axisSet = new Set(axis);
		if (axisSet.size !== axis.length) throw new TypeError("Gather axes are not unique");
		const gatherShape = indices.reduce((shape$1, a) => require_backend.generalBroadcast(shape$1, a.shape), []);
		const newShape = x.shape.filter((_, i) => !axisSet.has(i));
		newShape.splice(outDim, 0, ...gatherShape);
		return [new ShapedArray(newShape, x.dtype, x.weakType)];
	},
	[Primitive.Transpose]([x], { perm }) {
		return [new ShapedArray(perm.map((i) => x.shape[i]), x.dtype, x.weakType)];
	},
	[Primitive.Broadcast]([x], { shape: shape$1 }) {
		return [new ShapedArray(shape$1, x.dtype, x.weakType)];
	},
	[Primitive.Reshape]([x], { shape: shape$1 }) {
		return [new ShapedArray(shape$1, x.dtype, x.weakType)];
	},
	[Primitive.Flip]([x], _) {
		return [ShapedArray.fromAval(x)];
	},
	[Primitive.Shrink]([x], { slice }) {
		const newShape = slice.map((s) => s[1] - s[0]);
		return [new ShapedArray(newShape, x.dtype, x.weakType)];
	},
	[Primitive.Pad]([x], { width }) {
		const newShape = x.shape.map((dim, i) => dim + width[i][0] + width[i][1]);
		return [new ShapedArray(newShape, x.dtype, x.weakType)];
	},
	[Primitive.DynamicUpdateSlice]([dst, src], { offset, axis }) {
		if (!(dst instanceof ShapedArray) || !(src instanceof ShapedArray)) throw new TypeError("dynamicUpdateSlice expects shaped array inputs");
		const dstShape = dst.shape;
		const srcShape = src.shape;
		if (dstShape.length === srcShape.length) {
			for (let i = 0; i < dstShape.length; i++) {
				if (i === axis) continue;
				if (dstShape[i] !== srcShape[i]) throw new TypeError("dynamicUpdateSlice: shape mismatch");
			}
			if (offset + srcShape[axis] > dstShape[axis]) throw new TypeError("dynamicUpdateSlice: out of bounds");
		} else if (axis === 0 && dstShape.length === srcShape.length + 1) {
			for (let i = 0; i < srcShape.length; i++) if (dstShape[i + 1] !== srcShape[i]) throw new TypeError("dynamicUpdateSlice: stacked shape mismatch");
			if (offset + 1 > dstShape[0]) throw new TypeError("dynamicUpdateSlice: stacked out of bounds");
		} else throw new TypeError("dynamicUpdateSlice: unsupported shapes");
		return [new ShapedArray(dst.shape, dst.dtype, dst.weakType)];
	},
	[Primitive.Sort]([x]) {
		if (x.ndim === 0) throw new TypeError("sort: requires at least 1D input");
		return [ShapedArray.fromAval(x)];
	},
	[Primitive.Argsort]([x]) {
		if (x.ndim === 0) throw new TypeError("argsort: requires at least 1D input");
		return [ShapedArray.fromAval(x), new ShapedArray(x.shape, require_backend.DType.Int32, false)];
	},
	[Primitive.TriangularSolve]([a, b]) {
		if (a.ndim < 2) throw new TypeError(`triangular_solve: a must be at least 2D, got ${a}`);
		if (b.ndim < 2) throw new TypeError(`triangular_solve: b must be at least 2D, got ${b}`);
		const [m, n] = a.shape.slice(-2);
		const [_batch, q] = b.shape.slice(-2);
		if (!require_backend.deepEqual(a.shape.slice(0, -2), b.shape.slice(0, -2)) || a.dtype !== b.dtype || m !== n || n !== q) throw new TypeError(`triangular_solve: mismatch ${a} vs ${b}`);
		return [new ShapedArray(b.shape, b.dtype, a.weakType && b.weakType)];
	},
	[Primitive.Cholesky]([a]) {
		if (a.ndim < 2) throw new TypeError(`cholesky: requires at least 2D input, got ${a}`);
		if (a.shape[a.ndim - 2] !== a.shape[a.ndim - 1]) throw new TypeError(`cholesky: must be square, got ${a}`);
		return [ShapedArray.fromAval(a)];
	},
	[Primitive.LU]([a]) {
		if (a.ndim < 2) throw new TypeError(`lu: requires at least 2D input, got ${a}`);
		const batch = a.shape.slice(0, -2);
		const [m, n] = a.shape.slice(-2);
		return [
			ShapedArray.fromAval(a),
			new ShapedArray([...batch, Math.min(m, n)], require_backend.DType.Int32, false),
			new ShapedArray([...batch, m], require_backend.DType.Int32, false)
		];
	},
	[Primitive.Jit](args, { jaxpr }) {
		const { inTypes, outTypes } = typecheckJaxpr(jaxpr);
		if (args.length !== inTypes.length) throw new TypeError(`jit expected ${inTypes.length} arguments, got ${args.length}`);
		for (let i = 0; i < inTypes.length; i++) if (!args[i].equals(inTypes[i])) throw new TypeError(`jit argument ${i} has type ${args[i]}, expected ${inTypes[i]}`);
		return outTypes;
	},
	[Primitive.Scan](args, { jaxpr, numCarry, numConsts, length, reverse: _ }) {
		const numX = args.length - numConsts - numCarry;
		const { outTypes } = typecheckJaxpr(jaxpr);
		if (jaxpr.inBinders.length !== numConsts + numCarry + numX) throw new TypeError(`Scan jaxpr expects ${jaxpr.inBinders.length} inputs, got ${numConsts + numCarry + numX}`);
		const carryOutTypes = outTypes.slice(0, numCarry);
		const ySliceTypes = outTypes.slice(numCarry);
		const yTypes = ySliceTypes.map((t) => {
			return new ShapedArray([length, ...t.shape], t.dtype, t.weakType);
		});
		return [...carryOutTypes, ...yTypes];
	}
};
function splitIdx(values, argnums) {
	const a = [];
	const b = [];
	for (let i = 0; i < values.length; i++) if (argnums.has(i)) a.push(values[i]);
	else b.push(values[i]);
	return [a, b];
}
function joinIdx(n, a, b, argnums) {
	const result = [];
	let ai = 0;
	let bi = 0;
	for (let i = 0; i < n; i++) if (argnums.has(i)) result.push(a[ai++]);
	else result.push(b[bi++]);
	return result;
}
function makeJaxpr$1(f, opts) {
	return (...argsIn) => {
		try {
			var _usingCtx$1 = (0, import_usingCtx$11.default)();
			const staticArgnums = new Set(opts?.staticArgnums ?? []);
			const [staticArgs, shapedArgs] = splitIdx(argsIn, staticArgnums);
			const [avalsIn, inTree] = flatten(shapedArgs);
			const [fFlat, outTree] = flattenFun((...dynamicArgs) => {
				return f(...joinIdx(argsIn.length, staticArgs, dynamicArgs, staticArgnums));
			}, inTree);
			const builder = new JaxprBuilder();
			const main = _usingCtx$1.u(newMain(JaxprTrace, builder));
			main.isAbstract = true;
			_usingCtx$1.u(newDynamic(main));
			const trace$1 = new JaxprTrace(main);
			const tracersIn = avalsIn.map((aval) => trace$1.newArg(typeof aval === "object" ? aval : pureArray(aval)));
			const prevTracker = _peArrayCreationTracker;
			_setPACT(null);
			let outs;
			try {
				outs = fFlat(...tracersIn);
			} finally {
				_setPACT(prevTracker);
			}
			const tracersOut = outs.map((out) => fullRaise(trace$1, out));
			const jaxpr = builder.build(tracersIn, tracersOut);
			if (outTree.value === void 0) throw new Error("outTree was not set in makeJaxpr");
			return {
				jaxpr: jaxpr.mapJaxpr((j) => j.simplify()),
				treedef: outTree.value
			};
		} catch (_) {
			_usingCtx$1.e = _;
		} finally {
			_usingCtx$1.d();
		}
	};
}
function jit$1(f, opts) {
	const cache = /* @__PURE__ */ new Map();
	const staticArgnums = new Set(opts?.staticArgnums ?? []);
	const result = ((...args) => {
		const [staticArgs, dynamicArgs] = splitIdx(args, staticArgnums);
		const [argsFlat, inTree] = flatten(dynamicArgs);
		const avalsInFlat = argsFlat.map((x) => ShapedArray.fromAval(getAval(x)));
		const avalsIn = unflatten(inTree, avalsInFlat);
		const jaxprArgs = joinIdx(args.length, staticArgs, avalsIn, staticArgnums);
		const { jaxpr, treedef: outTree } = require_backend.runWithCache(cache, jaxprArgs, () => {
			return makeJaxpr$1(f, opts)(...jaxprArgs);
		});
		const outs = bind(Primitive.Jit, [...jaxpr.consts, ...argsFlat], {
			name: f.name || "closure",
			jaxpr: jaxpr.jaxpr,
			numConsts: jaxpr.consts.length
		});
		return unflatten(outTree, outs);
	});
	result.dispose = () => {
		for (const { jaxpr } of cache.values()) jaxpr.dispose();
		cache.clear();
	};
	result[Symbol.dispose] = result.dispose;
	_jitFunctionDisposers.add(result.dispose);
	return result;
}

//#endregion
//#region src/frontend/scan-executor.ts
/**
* Execute a scan loop. Dispatches to the appropriate strategy based on the plan.
*
* Ownership contract:
* - constSlots: borrowed (incRef'd before each body call, not consumed)
* - initCarrySlots: consumed (absorbed into first iteration)
* - xsSlots: borrowed (sliced per iteration via createView or readSync)
* - outputSlots: filled in-place (carry outputs + stacked Y outputs)
*/
function executeScan(params) {
	switch (params.plan.path) {
		case "fallback": return executeScanFallback(params);
		case "compiled-loop": return executeScanCompiledLoop(params);
		case "preencoded-routine": return executeScanPreencodedRoutine(params);
	}
}
function executeScanFallback(params) {
	const { backend, bodyProgram, bodyJaxpr, length, numCarry, numConsts: _numConsts, numX: _numX, numY, reverse, constSlots, initCarrySlots, xsSlots, xsAvals, outputSlots } = params;
	const xsStrides = xsAvals.map((aval) => aval.size * require_backend.byteWidth(aval.dtype));
	const yOutAvals = bodyJaxpr.outs.slice(numCarry).map((v) => v.aval);
	const ysStrides = yOutAvals.map((aval) => aval.size * require_backend.byteWidth(aval.dtype));
	let carry = initCarrySlots.slice();
	for (const slot of carry) backend.incRef(slot);
	const ysOutputSlots = outputSlots.slice(numCarry);
	const pending = [];
	for (let step = 0; step < length; step++) {
		const i = reverse ? length - 1 - step : step;
		flushPending(pending);
		const xSlices = sliceXsAtIteration(backend, xsSlots, xsStrides, xsAvals, i);
		for (const slot of constSlots) backend.incRef(slot);
		const bodyInputs = [
			...constSlots,
			...carry,
			...xSlices
		];
		const bodyResult = bodyProgram.execute(bodyInputs);
		pending.push(...bodyResult.pending);
		flushPending(pending);
		const newCarry = bodyResult.outputs.slice(0, numCarry);
		const ySlices = bodyResult.outputs.slice(numCarry);
		for (const slot of constSlots) backend.decRef(slot);
		for (const slot of xSlices) backend.decRef(slot);
		for (let yi = 0; yi < numY; yi++) {
			if (ysStrides[yi] > 0) copySliceToBuffer(backend, ysOutputSlots[yi], ySlices[yi], i, ysStrides[yi], ysStrides[yi]);
			backend.decRef(ySlices[yi]);
		}
		for (const slot of carry) backend.decRef(slot);
		carry = newCarry;
	}
	flushPending(pending);
	const carryOutputSlots = outputSlots.slice(0, numCarry);
	for (let ci = 0; ci < numCarry; ci++) {
		if (carry[ci] === carryOutputSlots[ci]) continue;
		const carrySize = bodyJaxpr.outs[ci].aval.size * require_backend.byteWidth(bodyJaxpr.outs[ci].aval.dtype);
		copySliceToBuffer(backend, carryOutputSlots[ci], carry[ci], 0, 0, carrySize);
		backend.decRef(carry[ci]);
	}
	return {
		outputs: outputSlots,
		pending
	};
}
function executeScanCompiledLoop(params) {
	const { backend, plan, numCarry, numY, constSlots, initCarrySlots, xsSlots, outputSlots } = params;
	if (plan.path !== "compiled-loop") throw new Error("unreachable");
	if (!plan.params) throw new Error("compiled-loop plan missing params");
	const carryOutSlots = outputSlots.slice(0, numCarry);
	const ysStackedSlots = outputSlots.slice(numCarry, numCarry + numY);
	if (backend.type === "webgpu") {
		const webgpuBackend = backend;
		webgpuBackend.dispatchNativeScanGeneral(plan.executable, plan.params, constSlots, initCarrySlots, xsSlots, carryOutSlots, ysStackedSlots);
	} else {
		const wasmBackend = backend;
		wasmBackend.dispatchNativeScanGeneral(plan.executable, plan.params, constSlots, initCarrySlots, xsSlots, carryOutSlots, ysStackedSlots);
	}
	return {
		outputs: outputSlots,
		pending: []
	};
}
function executeScanPreencodedRoutine(params) {
	const { backend, plan, numCarry, numY, constSlots, initCarrySlots, xsSlots, outputSlots } = params;
	if (plan.path !== "preencoded-routine") throw new Error("unreachable");
	const carryOutSlots = outputSlots.slice(0, numCarry);
	const ysStackedSlots = outputSlots.slice(numCarry, numCarry + numY);
	const webgpuBackend = backend;
	webgpuBackend.dispatchPreencodedScan(plan.preencodedParams, constSlots, initCarrySlots, xsSlots, carryOutSlots, ysStackedSlots);
	return {
		outputs: outputSlots,
		pending: []
	};
}
/** Flush all pending GPU/WASM operations. */
function flushPending(pending) {
	for (const p of pending) {
		p.prepareSync();
		p.submit();
	}
	pending.length = 0;
}
/**
* Slice xs buffers at a given iteration index.
* Returns new slots (views or copies) for each xs input.
*/
function sliceXsAtIteration(backend, xsSlots, xsStrides, xsAvals, iterIdx) {
	const slices = [];
	for (let j = 0; j < xsSlots.length; j++) {
		const srcOffset = iterIdx * xsStrides[j];
		const sliceSize = xsAvals[j].size * require_backend.byteWidth(xsAvals[j].dtype);
		if (backend.copyBufferToBuffer) {
			const slot = backend.malloc(sliceSize);
			backend.copyBufferToBuffer(xsSlots[j], srcOffset, slot, 0, sliceSize);
			slices.push(slot);
		} else {
			const data = backend.readSync(xsSlots[j], srcOffset, sliceSize);
			const slot = backend.malloc(sliceSize, data);
			slices.push(slot);
		}
	}
	return slices;
}
/**
* Copy a slice from src slot into dst slot at a given iteration offset.
*/
function copySliceToBuffer(backend, dst, src, iterIdx, strideBytes, sliceBytes) {
	const dstOffset = iterIdx * strideBytes;
	if (backend.copyBufferToBuffer) backend.copyBufferToBuffer(src, 0, dst, dstOffset, sliceBytes);
	else {
		const data = backend.readSync(src, 0, sliceBytes);
		const dstData = backend.readSync(dst);
		const dstArray = new Uint8Array(dstData.buffer.slice(0));
		dstArray.set(data, dstOffset);
		throw new Error("internal: copySliceToBuffer requires backend.copyBufferToBuffer");
	}
}

//#endregion
//#region src/frontend/scan-plan.ts
/**
* Check if a chosen scan path satisfies the acceptPath constraint.
* Returns an error message if the path is not allowed, or null if OK.
*
* Special case: an empty array `[]` always rejects, showing the chosen path.
*/
function checkAcceptedPath(chosenPath, acceptPath, extraInfo) {
	if (!acceptPath) return null;
	const allowedPaths = Array.isArray(acceptPath) ? acceptPath : [acceptPath];
	const suffix = extraInfo ? ` (${extraInfo})` : "";
	if (allowedPaths.length === 0) return `Scan path debug: chose "${chosenPath}"${suffix}`;
	if (!allowedPaths.includes(chosenPath)) return `Scan acceptPath constraint not satisfied: got "${chosenPath}" but accepted paths are [${allowedPaths.map((p) => `"${p}"`).join(", ")}]${suffix}`;
	return null;
}
/**
* Extract buffer sizes and strides from body jaxpr for native scan codegen.
* Shared by WebGPU and WASM native scan implementations.
*/
function getScanBufferSizes(bodyJaxpr, numConsts, numCarry, numX) {
	const constAvals = bodyJaxpr.inBinders.slice(0, numConsts).map((v) => v.aval);
	const carryAvals = bodyJaxpr.inBinders.slice(numConsts, numConsts + numCarry).map((v) => v.aval);
	const xAvals = bodyJaxpr.inBinders.slice(numConsts + numCarry, numConsts + numCarry + numX).map((v) => v.aval);
	const yAvals = bodyJaxpr.outs.slice(numCarry).map((v) => v.aval);
	return {
		constSizes: constAvals.map((a) => a.size * require_backend.byteWidth(a.dtype)),
		carrySizes: carryAvals.map((a) => a.size * require_backend.byteWidth(a.dtype)),
		xsStrides: xAvals.map((a) => a.size * require_backend.byteWidth(a.dtype)),
		ysStrides: yAvals.map((a) => a.size * require_backend.byteWidth(a.dtype))
	};
}
/**
* Try to prepare a WASM native scan executable.
*
* Builds GeneralScanStep[] from the body program's execute steps, maps
* slot IDs to internal buffer indices, determines CarryOutputSource and
* YOutputSource for each output, and calls backend.prepareNativeScanGeneral().
*
* Returns null if the body can't be compiled (e.g. unsupported step types,
* unmapped slots).
*/
function tryPrepareWasmNativeScan(backend, bodyProgram, bodyJaxpr, executeSteps, length, numCarry, numConsts, numX, numY, reverse) {
	if (require_backend.DEBUG >= 2) console.log(`[wasm-scan] trying with numCarry=${numCarry}, numY=${numY}, steps=${executeSteps.length}`);
	for (const step of executeSteps) if (step.source instanceof require_backend.Routine) {
		if (!require_backend.getScanRoutineInfo(step.source)) {
			if (require_backend.DEBUG >= 1) console.log(`[wasm-scan] skipped, unsupported routine: ${step.source.name}`);
			return null;
		}
	}
	const numInputs = numConsts + numCarry + numX;
	const { constSizes, carrySizes, xsStrides, ysStrides } = getScanBufferSizes(bodyJaxpr, numConsts, numCarry, numX);
	const slotToInternal = /* @__PURE__ */ new Map();
	const stepToInternalBase = /* @__PURE__ */ new Map();
	const internalSizes = [];
	for (let i = 0; i < executeSteps.length; i++) {
		const step = executeSteps[i];
		const source = step.source;
		stepToInternalBase.set(i, internalSizes.length);
		if (source instanceof require_backend.Kernel) {
			const internalIdx = internalSizes.length;
			slotToInternal.set(step.outputs[0], internalIdx);
			internalSizes.push(source.size * require_backend.byteWidth(source.dtype));
		} else if (source instanceof require_backend.Routine) for (let outIdx = 0; outIdx < step.outputs.length; outIdx++) {
			const internalIdx = internalSizes.length;
			slotToInternal.set(step.outputs[outIdx], internalIdx);
			const outShape = source.type.outputShapes[outIdx];
			const outDtype = source.type.outputDtypes[outIdx];
			internalSizes.push(outShape.reduce((a, b) => a * b, 1) * require_backend.byteWidth(outDtype));
		}
	}
	let auxBufferSize = 0;
	let elementSize = 4;
	for (const step of executeSteps) if (step.source instanceof require_backend.Routine) {
		const routine = step.source;
		const dtype = routine.type.inputDtypes[0];
		elementSize = require_backend.byteWidth(dtype);
		if (routine.name === require_backend.Routines.Sort) {
			const inputShape = routine.type.inputShapes[0];
			const sortDim = inputShape[inputShape.length - 1];
			auxBufferSize = Math.max(auxBufferSize, sortDim * elementSize);
		} else if (routine.name === require_backend.Routines.Argsort) {
			const inputShape = routine.type.inputShapes[0];
			const sortDim = inputShape[inputShape.length - 1];
			auxBufferSize = Math.max(auxBufferSize, sortDim * 4);
		}
	}
	const routineInfos = [];
	const stepToRoutineInfoIdx = /* @__PURE__ */ new Map();
	for (let i = 0; i < executeSteps.length; i++) {
		const step = executeSteps[i];
		if (step.source instanceof require_backend.Routine) {
			const routine = step.source;
			const info = require_backend.getScanRoutineInfo(routine);
			if (!info) throw new Error(`[wasm-scan] Unexpected unsupported routine: ${routine.name}`);
			const routineInfoIdx = routineInfos.length;
			stepToRoutineInfoIdx.set(i, routineInfoIdx);
			routineInfos.push(info);
		}
	}
	const steps = [];
	for (let i = 0; i < executeSteps.length; i++) {
		const step = executeSteps[i];
		const source = step.source;
		const inputSlots = [];
		for (const inputId of step.inputs) if (inputId < numInputs) inputSlots.push(inputId);
		else {
			const internalIdx = slotToInternal.get(inputId);
			if (internalIdx === void 0) {
				if (require_backend.DEBUG >= 1) console.log(`[wasm-scan] skipped, input ${inputId} not found in slot mapping`);
				return null;
			}
			inputSlots.push(numInputs + internalIdx);
		}
		if (source instanceof require_backend.Kernel) {
			const reindexMap = inputSlots;
			const reindexedExp = source.exp.reindexGids(reindexMap);
			const reindexedReduction = source.reduction ? new require_backend.Reduction(source.reduction.dtype, source.reduction.op, source.reduction.size, source.reduction.epilogue.reindexGids(reindexMap)) : void 0;
			const reindexedKernel = new require_backend.Kernel(numInputs + internalSizes.length, source.size, reindexedExp, reindexedReduction);
			const internalBase = stepToInternalBase.get(i);
			steps.push({
				source: reindexedKernel,
				inputSlots,
				outputInternalIdx: internalBase
			});
		} else if (source instanceof require_backend.Routine) {
			const routine = source;
			const routineName = routine.name;
			const routineInfoIdx = stepToRoutineInfoIdx.get(i);
			const internalBase = stepToInternalBase.get(i);
			const numOutputs = routine.type.outputShapes.length;
			const outputInternalIndices = [];
			for (let outIdx = 0; outIdx < numOutputs; outIdx++) outputInternalIndices.push(internalBase + outIdx);
			let staticParams = [];
			if (routineName === require_backend.Routines.Cholesky) {
				const inputShape = routine.type.inputShapes[0];
				staticParams = [inputShape[inputShape.length - 1]];
			} else if (routineName === require_backend.Routines.Sort) {
				const inputShape = routine.type.inputShapes[0];
				staticParams = [inputShape[inputShape.length - 1]];
			} else if (routineName === require_backend.Routines.TriangularSolve) {
				const aShape = routine.type.inputShapes[0];
				const bShape = routine.type.inputShapes[1];
				const n = aShape[aShape.length - 1];
				const batchRows = bShape[bShape.length - 1];
				const numBatches = 1;
				const unitDiagonal = routine.params?.unitDiagonal ? 1 : 0;
				const lower = 0;
				staticParams = [
					n,
					batchRows,
					numBatches,
					unitDiagonal,
					lower
				];
			} else if (routineName === require_backend.Routines.LU) {
				const inputShape = routine.type.inputShapes[0];
				const m = inputShape[inputShape.length - 2];
				const n = inputShape[inputShape.length - 1];
				staticParams = [m, n];
			} else if (routineName === require_backend.Routines.Argsort) {
				const inputShape = routine.type.inputShapes[0];
				staticParams = [inputShape[inputShape.length - 1]];
			}
			steps.push({
				source,
				inputSlots,
				outputInternalIdx: internalBase,
				outputInternalIndices,
				routineCallInfo: {
					routineInfoIdx,
					staticParams
				}
			});
		}
	}
	const carryOutSlots = bodyProgram.outputs.slice(0, numCarry);
	const carryInputSlots = bodyProgram.inputs.slice(numConsts, numConsts + numCarry);
	const carryOutSources = [];
	for (const slot of carryOutSlots) {
		const carryIdx = carryInputSlots.indexOf(slot);
		if (carryIdx !== -1) {
			carryOutSources.push({
				type: "passthrough",
				carryIdx
			});
			continue;
		}
		const internalIdx = slotToInternal.get(slot);
		if (internalIdx === void 0) {
			if (require_backend.DEBUG >= 1) console.log(`[wasm-scan] skipped, carry output slot ${slot} not produced by any execute step`);
			return null;
		}
		carryOutSources.push({
			type: "internal",
			internalIdx
		});
	}
	const xsInputSlots = bodyProgram.inputs.slice(numConsts + numCarry, numConsts + numCarry + numX);
	const yOutputSlots = bodyProgram.outputs.slice(numCarry);
	const yOutputSources = [];
	for (const slot of yOutputSlots) {
		const carryIdx = carryInputSlots.indexOf(slot);
		if (carryIdx !== -1) {
			yOutputSources.push({
				type: "passthrough",
				carryIdx
			});
			continue;
		}
		const xsIdx = xsInputSlots.indexOf(slot);
		if (xsIdx !== -1) {
			yOutputSources.push({
				type: "xs-passthrough",
				xsIdx
			});
			continue;
		}
		const internalIdx = slotToInternal.get(slot);
		if (internalIdx === void 0) {
			if (require_backend.DEBUG >= 1) console.log(`[wasm-scan] skipped, Y output slot ${slot} not found`);
			return null;
		}
		yOutputSources.push({
			type: "internal",
			internalIdx
		});
	}
	const wasmBackend = backend;
	if (!wasmBackend.prepareNativeScanGeneral) {
		if (require_backend.DEBUG >= 2) console.log("[wasm-scan] backend has no prepareNativeScanGeneral");
		return null;
	}
	const params = {
		length,
		numConsts,
		constSizes,
		numCarry,
		carrySizes,
		numX,
		xsStrides,
		numY,
		ysStrides,
		internalSizes,
		steps,
		carryOutSources,
		yOutputSources,
		reverse,
		auxBufferSize: auxBufferSize > 0 ? auxBufferSize : void 0,
		elementSize: auxBufferSize > 0 ? elementSize : void 0,
		routineInfos: routineInfos.length > 0 ? routineInfos : void 0
	};
	try {
		const exe = wasmBackend.prepareNativeScanGeneral(params);
		if (exe) {
			if (require_backend.DEBUG >= 1) {
				const hasRoutines = steps.some((s) => s.source instanceof require_backend.Routine);
				console.log(`[wasm-scan] SUCCESS! Using WASM native scan with ${steps.length} steps` + (hasRoutines ? " (includes routines)" : ""));
			}
			return {
				executable: exe,
				internalSizes,
				params
			};
		}
		return null;
	} catch (e$1) {
		if (require_backend.DEBUG >= 2) console.warn("[wasm-scan] preparation failed:", e$1);
		return null;
	}
}
/**
* Try to prepare a WebGPU native scan.
*
* Constraints:
* - numCarry === numY or numY === 0 (each carry maps 1:1 to a Y output)
* - No routine steps (only kernel steps)
* - No passthrough carry (each carry output must come from a kernel step)
* - No internal buffer dependencies (step inputs reference jaxpr inputs only)
*/
function tryPrepareWebGPUNativeScan(backend, bodyProgram, bodyJaxpr, executeSteps, length, numCarry, numConsts, numX, numY, reverse) {
	if (numY !== 0 && numCarry !== numY) {
		if (require_backend.DEBUG >= 1) console.log(`[webgpu-scan] skipped, numCarry=${numCarry} !== numY=${numY}`);
		return null;
	}
	for (const step of executeSteps) if (step.source instanceof require_backend.Routine) {
		if (require_backend.DEBUG >= 1) console.log(`[webgpu-scan] skipped, routine in scan body`);
		return null;
	}
	const numInputs = numConsts + numCarry + numX;
	const { constSizes, carrySizes, xsStrides, ysStrides } = getScanBufferSizes(bodyJaxpr, numConsts, numCarry, numX);
	const outputToStep = /* @__PURE__ */ new Map();
	for (const step of executeSteps) outputToStep.set(step.outputs[0], step);
	const carryOutIds = bodyProgram.outputs.slice(0, numCarry);
	const carryInputIds = bodyProgram.inputs.slice(numConsts, numConsts + numCarry);
	const stepToCarryIdx = /* @__PURE__ */ new Map();
	for (let ci = 0; ci < numCarry; ci++) {
		const outId = carryOutIds[ci];
		if (carryInputIds.includes(outId)) {
			if (require_backend.DEBUG >= 1) console.log(`[webgpu-scan] skipped, carry ${ci} is passthrough`);
			return null;
		}
		const step = outputToStep.get(outId);
		if (!step) {
			if (require_backend.DEBUG >= 1) console.log(`[webgpu-scan] skipped, carry ${ci} not produced by execute step`);
			return null;
		}
		stepToCarryIdx.set(step, ci);
	}
	if (numY > 0) {
		const yOutIds = bodyProgram.outputs.slice(numCarry);
		for (let yi = 0; yi < numY; yi++) if (yOutIds[yi] !== carryOutIds[yi]) {
			if (require_backend.DEBUG >= 1) console.log(`[webgpu-scan] skipped, y${yi} output slot differs from carry${yi}`);
			return null;
		}
	}
	for (const step of executeSteps) for (const inputId of step.inputs) if (inputId >= numInputs) {
		if (require_backend.DEBUG >= 1) console.log(`[webgpu-scan] skipped, step has internal dep (input ${inputId} >= ${numInputs})`);
		return null;
	}
	const orderedSteps = [];
	for (const [step, carryIdx] of stepToCarryIdx) orderedSteps.push({
		carryIdx,
		step
	});
	orderedSteps.sort((a, b) => a.carryIdx - b.carryIdx);
	for (let i = 0; i < orderedSteps.length; i++) {
		const writtenCarryGid = numConsts + orderedSteps[i].carryIdx;
		for (let j = i + 1; j < orderedSteps.length; j++) {
			const laterStep = orderedSteps[j].step;
			const laterSource = laterStep.source;
			const laterReindexMap = laterStep.inputs;
			const laterExp = laterSource.exp.reindexGids(laterReindexMap);
			const readsWrittenCarry = laterExp.some((e$1) => (e$1.op === require_backend.AluOp.GlobalIndex || e$1.op === require_backend.AluOp.GlobalView) && e$1.arg[0] === writtenCarryGid);
			if (readsWrittenCarry) {
				if (require_backend.DEBUG >= 1) console.log(`[webgpu-scan] skipped, step ${j} reads carry[${orderedSteps[i].carryIdx}] written by step ${i}`);
				return null;
			}
			if (laterSource.reduction) {
				const laterReductionExp = laterSource.reduction.epilogue.reindexGids(laterReindexMap);
				const readsInReduction = laterReductionExp.some((e$1) => (e$1.op === require_backend.AluOp.GlobalIndex || e$1.op === require_backend.AluOp.GlobalView) && e$1.arg[0] === writtenCarryGid);
				if (readsInReduction) {
					if (require_backend.DEBUG >= 1) console.log(`[webgpu-scan] skipped, step ${j} reduction reads carry[${orderedSteps[i].carryIdx}] written by step ${i}`);
					return null;
				}
			}
		}
	}
	const multiSteps = orderedSteps.map(({ carryIdx, step }) => {
		const source = step.source;
		const reindexMap = step.inputs;
		const reindexedExp = source.exp.reindexGids(reindexMap);
		const reindexedReduction = source.reduction ? new require_backend.Reduction(source.reduction.dtype, source.reduction.op, source.reduction.size, source.reduction.epilogue.reindexGids(reindexMap)) : void 0;
		const reindexedKernel = new require_backend.Kernel(numInputs, source.size, reindexedExp, reindexedReduction);
		return {
			kernel: reindexedKernel,
			inputs: reindexMap.slice(),
			outputCarryIdx: carryIdx,
			outputSize: source.size
		};
	});
	const params = {
		length,
		numConsts,
		constSizes,
		numCarry,
		carrySizes,
		numX,
		xsStrides,
		numY,
		ysStrides,
		steps: multiSteps,
		reverse
	};
	const webgpuBackend = backend;
	const exe = webgpuBackend.prepareNativeScanMulti(params);
	if (!exe) return null;
	if (require_backend.DEBUG >= 1) console.log(`[webgpu-scan] SUCCESS! Using WebGPU native scan with ${multiSteps.length} steps`);
	return {
		executable: exe,
		params
	};
}
/**
* Try to prepare a native scan executable (any backend).
*/
function tryPrepareNativeScan(backend, bodyProgram, bodyJaxpr, length, numCarry, numConsts, numX, numY, reverse) {
	const executeSteps = bodyProgram.steps.filter((s) => s.type === "execute");
	if (executeSteps.length === 0) {
		if (require_backend.DEBUG >= 1) console.log("[compiled-loop] skipped, no execute steps");
		return null;
	}
	if (backend.type === "wasm") return tryPrepareWasmNativeScan(backend, bodyProgram, bodyJaxpr, executeSteps, length, numCarry, numConsts, numX, numY, reverse);
	if (backend.type === "webgpu") return tryPrepareWebGPUNativeScan(backend, bodyProgram, bodyJaxpr, executeSteps, length, numCarry, numConsts, numX, numY, reverse);
	if (require_backend.DEBUG >= 1) console.log(`[compiled-loop] skipped, backend=${backend.type} not supported yet`);
	return null;
}
/**
* Try to prepare a preencoded scan for routine bodies (matmul, cholesky, etc.).
*
* Requirements:
* - WebGPU backend
* - Exactly 1 execute step in body that is a Routine
* - numCarry === numY (passthrough pattern)
* - Routine shader must not already use uniforms (e.g. Sort excluded)
*/
function tryPreparePreencodedScan(backend, bodyProgram, bodyJaxpr, length, numCarry, numConsts, numX, numY, reverse) {
	if (backend.type !== "webgpu") {
		if (require_backend.DEBUG >= 2) console.log("Preencoded scan: skipped, unsupported backend");
		return null;
	}
	const executeSteps = bodyProgram.steps.filter((s) => s.type === "execute");
	if (executeSteps.length !== 1) {
		if (require_backend.DEBUG >= 2) console.log(`Preencoded scan: skipped, ${executeSteps.length} execute steps (need exactly 1)`);
		return null;
	}
	const execStep = executeSteps[0];
	if (!(execStep.source instanceof require_backend.Routine)) {
		if (require_backend.DEBUG >= 2) console.log("Preencoded scan: skipped, not a Routine");
		return null;
	}
	if (numCarry !== numY) {
		if (require_backend.DEBUG >= 2) console.log(`Preencoded scan: skipped, numCarry=${numCarry} !== numY=${numY}`);
		return null;
	}
	const carryAvals = bodyJaxpr.inBinders.slice(numConsts, numConsts + numCarry).map((v) => v.aval);
	const xAvals = bodyJaxpr.inBinders.slice(numConsts + numCarry).map((v) => v.aval);
	const carrySizes = carryAvals.map((a) => a.size * require_backend.byteWidth(a.dtype));
	const xsElemStrides = xAvals.map((a) => a.size);
	const ysElemStrides = carryAvals.map((a) => a.size);
	if (!backend.prepareRoutineSync) {
		if (require_backend.DEBUG >= 2) console.log("Preencoded scan: skipped, backend has no prepareRoutineSync");
		return null;
	}
	let bodyRoutineExe;
	try {
		bodyRoutineExe = backend.prepareRoutineSync(execStep.source);
	} catch (e$1) {
		if (require_backend.DEBUG >= 2) console.warn("Preencoded scan: prepareRoutineSync failed:", e$1);
		return null;
	}
	const webgpuBackend = backend;
	if (!webgpuBackend.preparePreencodedScan) {
		if (require_backend.DEBUG >= 2) console.log("Preencoded scan: skipped, backend has no preparePreencodedScan");
		return null;
	}
	const preencodedScanParams = {
		length,
		carrySizes,
		xsElemStrides,
		ysElemStrides,
		bodyRoutine: bodyRoutineExe,
		numCarry,
		numX,
		numY,
		numConsts,
		reverse,
		routineInputJitIds: execStep.inputs,
		routineOutputJitIds: execStep.outputs
	};
	try {
		const prepared = webgpuBackend.preparePreencodedScan(preencodedScanParams);
		if (prepared && require_backend.DEBUG >= 1) console.log(`Preencoded scan: SUCCESS! Using WebGPU preencoded scan for ${execStep.source.name}`);
		return prepared;
	} catch (e$1) {
		if (require_backend.DEBUG >= 2) console.warn("Preencoded scan preparation failed:", e$1);
		return null;
	}
}
/**
* Choose a scan execution strategy.
*
* Priority: compiled-loop > preencoded-routine > fallback.
*/
function planScan(backend, bodyProgram, bodyJaxpr, length, numCarry, numConsts, numX, numY, reverse, acceptPath) {
	const nativeScanResult = tryPrepareNativeScan(backend, bodyProgram, bodyJaxpr, length, numCarry, numConsts, numX, numY, reverse);
	if (nativeScanResult) {
		const pathError$1 = checkAcceptedPath("compiled-loop", acceptPath);
		if (pathError$1) throw new Error(pathError$1);
		return {
			path: "compiled-loop",
			executable: nativeScanResult.executable,
			internalSizes: nativeScanResult.internalSizes,
			params: nativeScanResult.params
		};
	}
	const preencodedResult = tryPreparePreencodedScan(backend, bodyProgram, bodyJaxpr, length, numCarry, numConsts, numX, numY, reverse);
	if (preencodedResult) {
		const pathError$1 = checkAcceptedPath("preencoded-routine", acceptPath);
		if (pathError$1) throw new Error(pathError$1);
		return {
			path: "preencoded-routine",
			preencodedParams: preencodedResult
		};
	}
	const dispatchCount = bodyProgram.steps.filter((s) => s.type === "execute").length;
	const extraInfo = `${dispatchCount} dispatch${dispatchCount !== 1 ? "es" : ""} per iteration`;
	const pathError = checkAcceptedPath("fallback", acceptPath, extraInfo);
	if (pathError) throw new Error(pathError);
	return {
		path: "fallback",
		extraInfo
	};
}

//#endregion
//#region src/frontend/jit.ts
/** Walk JitSteps to compute peak live bytes and the set of malloc sizes. */
function computePoolHints(steps) {
	const mallocSizes = /* @__PURE__ */ new Set();
	const sizeOf = /* @__PURE__ */ new Map();
	for (const s of steps) if (s.type === "malloc") {
		const padded = Math.ceil(s.size / 4) * 4;
		sizeOf.set(s.output, padded);
		mallocSizes.add(padded);
	}
	let liveBytes = 0;
	let peakBytes = 0;
	const live = /* @__PURE__ */ new Map();
	for (const s of steps) {
		switch (s.type) {
			case "malloc": {
				const padded = sizeOf.get(s.output);
				live.set(s.output, padded);
				liveBytes += padded;
				break;
			}
			case "free":
				liveBytes -= live.get(s.input) ?? 0;
				live.delete(s.input);
				break;
			case "recycle":
				live.set(s.output, live.get(s.input) ?? 0);
				live.delete(s.input);
				break;
		}
		if (liveBytes > peakBytes) peakBytes = liveBytes;
	}
	return {
		peakBytes,
		mallocSizes
	};
}
/** Result of compiling a Jaxpr. Can be evaluated on a series of inputs. */
var JitProgram = class {
	poolHints;
	constructor(backend, steps, inputs, outputs) {
		this.backend = backend;
		this.steps = steps;
		this.inputs = inputs;
		this.outputs = outputs;
		this.poolHints = computePoolHints(steps);
	}
	pprint() {
		const steps = this.steps.map((step) => {
			switch (step.type) {
				case "execute": {
					const inputsNice = step.inputs.map((id, i) => `${i}: %${id}`).join(", ");
					const outputsNice = step.outputs.map((id) => `%${id}`).join(", ");
					const executeText = `execute (${inputsNice}) -> ${outputsNice}`;
					if (step.source instanceof require_backend.Kernel) return require_backend.PPrint.pp(`${executeText}, kernel`).concat(step.source.pprint().indent(2));
					else if (step.source instanceof require_backend.Routine) return require_backend.PPrint.pp(`${executeText}, routine ${step.source.name}`);
					else {
						step.source;
						return require_backend.PPrint.pp(executeText);
					}
				}
				case "malloc": return require_backend.PPrint.pp(`%${step.output} = malloc <${step.size} bytes>`);
				case "incref": return require_backend.PPrint.pp(`incref ${step.input}`);
				case "free": return require_backend.PPrint.pp(`free ${step.input}`);
				case "recycle": return require_backend.PPrint.pp(`%${step.output} = recycle %${step.input}`);
				case "scan": return require_backend.PPrint.pp(`scan [${step.plan.path}] length=${step.length} numCarry=${step.numCarry} numConsts=${step.numConsts} numX=${step.numX} numY=${step.numY}` + (step.reverse ? " reverse" : ""));
			}
		});
		const display = require_backend.PPrint.prototype.concat(require_backend.PPrint.pp(`device = ${this.backend.type}`), require_backend.PPrint.pp("inputs = [" + this.inputs.join(", ") + "]"), require_backend.PPrint.pp("outputs = [" + this.outputs.join(", ") + "]"), require_backend.PPrint.pp("steps ="), require_backend.PPrint.prototype.concat(...steps).indent(2));
		return require_backend.PPrint.pp("{ ").stack(display.stack(require_backend.PPrint.pp(" }")));
	}
	toString() {
		return this.pprint().toString();
	}
	/** Execute the JitProgram with the given inputs. */
	execute(inputs) {
		this.backend.configurePool?.(this.poolHints);
		const scope = /* @__PURE__ */ new Map();
		if (inputs.length !== this.inputs.length) throw new TypeError(`Expected ${this.inputs.length} inputs, got ${inputs.length}`);
		for (const [i, id] of this.inputs.entries()) scope.set(id, inputs[i]);
		const pending = [];
		for (const step of this.steps) switch (step.type) {
			case "execute": {
				const inputs$1 = step.inputs.map((id) => scope.get(id));
				const outputs = step.outputs.map((id) => scope.get(id));
				if (inputs$1.some((s) => s === void 0) || outputs.some((s) => s === void 0)) throw new Error(`internal: JitProgram scope undefined`);
				pending.push(new PendingExecute(this.backend, step.source, inputs$1, outputs));
				break;
			}
			case "malloc": {
				const slot = this.backend.malloc(step.size);
				scope.set(step.output, slot);
				break;
			}
			case "incref": {
				const slot = scope.get(step.input);
				this.backend.incRef(slot);
				break;
			}
			case "free": {
				const slot = scope.get(step.input);
				this.backend.decRef(slot);
				scope.delete(step.input);
				break;
			}
			case "recycle": {
				const slot = scope.get(step.input);
				scope.delete(step.input);
				scope.set(step.output, slot);
				break;
			}
			case "scan": {
				for (const p of pending) {
					p.prepareSync();
					p.submit();
				}
				pending.length = 0;
				const constSlots = step.consts.map((id) => scope.get(id));
				const initCarrySlots = step.initCarry.map((id) => scope.get(id));
				const xsSlots = step.xs.map((id) => scope.get(id));
				const outputSlots = step.outputs.map((id) => scope.get(id));
				for (const s of constSlots) this.backend.incRef(s);
				for (const s of xsSlots) this.backend.incRef(s);
				const result = executeScan({
					backend: this.backend,
					plan: step.plan,
					bodyProgram: step.bodyProgram,
					bodyJaxpr: step.bodyJaxpr,
					length: step.length,
					numCarry: step.numCarry,
					numConsts: step.numConsts,
					numX: step.numX,
					numY: step.numY,
					reverse: step.reverse,
					constSlots,
					initCarrySlots,
					xsSlots,
					xsAvals: step.xsAvals,
					outputSlots
				});
				for (const s of constSlots) this.backend.decRef(s);
				for (const s of xsSlots) this.backend.decRef(s);
				pending.push(...result.pending);
				for (let oi = 0; oi < step.outputs.length; oi++) scope.set(step.outputs[oi], result.outputs[oi]);
				break;
			}
			default:
		}
		return {
			outputs: this.outputs.map((id) => scope.get(id)),
			pending
		};
	}
};
var JitProgramBuilder = class {
	backend;
	#nextId;
	steps;
	constructor(backend, nargs) {
		this.backend = backend;
		this.#nextId = nargs;
		this.steps = [];
	}
	pushLit(lit) {
		const kernel = new require_backend.Kernel(0, lit.aval.size, require_backend.AluExp.const(lit.dtype, lit.value));
		return this.pushKernel(kernel, []);
	}
	pushBuffer(size$1) {
		const id = this.#nextId++;
		this.steps.push({
			type: "malloc",
			size: size$1,
			output: id
		});
		return id;
	}
	pushKernel(kernel, inputs) {
		const id = this.pushBuffer(kernel.bytes);
		this.steps.push({
			type: "execute",
			source: kernel,
			inputs,
			outputs: [id]
		});
		return id;
	}
	pushRoutine(routine, inputs, outputs) {
		this.steps.push({
			type: "execute",
			source: routine,
			inputs,
			outputs
		});
	}
	pushIncref(id) {
		this.steps.push({
			type: "incref",
			input: id
		});
	}
	insertFreeSteps(outputIds) {
		const ids = this.steps.filter((s) => s.type === "malloc").map((s) => s.output);
		for (const id of ids) {
			if (outputIds.includes(id)) continue;
			const lastUsage = this.steps.findLastIndex((s) => s.type === "execute" && (s.outputs.includes(id) || s.inputs.includes(id)) || s.type === "malloc" && s.output === id || s.type === "scan" && (s.outputs.includes(id) || s.consts.includes(id) || s.initCarry.includes(id) || s.xs.includes(id)));
			this.steps.splice(lastUsage + 1, 0, {
				type: "free",
				input: id
			});
		}
	}
	pushFree(id) {
		this.steps.push({
			type: "free",
			input: id
		});
	}
	/**
	* Replace consecutive free→malloc pairs of the same byte size with a single
	* "recycle" step, reusing the same backend Slot without any alloc/free calls.
	*
	* This is safe because:
	* 1. The free'd buffer's last consumer has already been scheduled
	* 2. The malloc'd buffer hasn't been written yet (execute comes after)
	* 3. Sizes match exactly, so no memory waste and no peak-memory increase
	*/
	recycleBuffers() {
		const mallocSizes = /* @__PURE__ */ new Map();
		for (const s of this.steps) if (s.type === "malloc") mallocSizes.set(s.output, s.size);
		let recycleCount = 0;
		for (let i = 0; i < this.steps.length - 1; i++) {
			const step = this.steps[i];
			if (step.type !== "free") continue;
			const freeSize = mallocSizes.get(step.input);
			if (freeSize === void 0 || freeSize === 0) continue;
			for (let j = i + 1; j < this.steps.length; j++) {
				const next = this.steps[j];
				if (next.type === "malloc" && next.size === freeSize) {
					this.steps.splice(i, 1, {
						type: "recycle",
						input: step.input,
						output: next.output
					});
					this.steps.splice(j, 1);
					recycleCount++;
					break;
				}
				if (next.type === "incref" || next.type === "free") continue;
				break;
			}
		}
		if (require_backend.DEBUG >= 1 && recycleCount > 0) console.info(`jit: recycled ${recycleCount} buffer(s)`);
	}
};
const jitCompileCache = /* @__PURE__ */ new Map();
/**
* Clear the internal JIT compilation cache. Called by `_disposeAllJitCaches()`
* in jaxpr.ts during leak checking.
* @internal
*/
function _clearJitCompileCache() {
	jitCompileCache.clear();
}
_registerJitCacheDisposer(_clearJitCompileCache);
function jitCompile(backend, jaxpr) {
	const cacheKey = backend.type + "," + require_backend.FpHash.hash(jaxpr);
	const cached = jitCompileCache.get(cacheKey);
	if (cached) return cached;
	if (require_backend.DEBUG >= 1) console.info("=========== JIT Compile ===========\n" + jaxpr.toString());
	jaxpr = jaxpr.flatten().simplify();
	const nargs = jaxpr.inBinders.length;
	const builder = new JitProgramBuilder(backend, nargs);
	const blackNodes = splitGraphDataflow(backend, jaxpr);
	const ctx = /* @__PURE__ */ new Map();
	for (let i = 0; i < nargs; i++) {
		const v = jaxpr.inBinders[i];
		ctx.set(v, {
			type: "imm",
			arg: i
		});
	}
	for (let i = 0; i < jaxpr.eqns.length; i++) {
		const eqn = jaxpr.eqns[i];
		if (eqn.primitive === Primitive.Scan) {
			const params = eqn.params;
			const { jaxpr: bodyJaxpr, numCarry, numConsts, length, reverse, acceptPath } = params;
			const numX = bodyJaxpr.inBinders.length - numConsts - numCarry;
			const numY = bodyJaxpr.outs.length - numCarry;
			const allInputIds = [];
			for (const input of eqn.inputs) if (input instanceof Var) {
				const jv = ctx.get(input);
				if (jv.type !== "imm") throw new Error("jit: scan primitive input is not imm");
				allInputIds.push(jv.arg);
			} else if (input instanceof Lit) allInputIds.push(builder.pushLit(input));
			const constsIds = allInputIds.slice(0, numConsts);
			const initCarryIds = allInputIds.slice(numConsts, numConsts + numCarry);
			const xsIds = allInputIds.slice(numConsts + numCarry);
			const xsAvals = [];
			const xsInputs = eqn.inputs.slice(numConsts + numCarry);
			for (const input of xsInputs) xsAvals.push(input.aval);
			const outputIds$1 = [];
			for (const outVar of eqn.outBinders) {
				const outId = builder.pushBuffer(outVar.aval.size * require_backend.byteWidth(outVar.aval.dtype));
				outputIds$1.push(outId);
				ctx.set(outVar, {
					type: "imm",
					arg: outId
				});
			}
			const bodyProgram = jitCompile(backend, bodyJaxpr);
			const scanPlan = planScan(backend, bodyProgram, bodyJaxpr, length, numCarry, numConsts, numX, numY, reverse, acceptPath);
			const xsSliceAvals = xsAvals.map((a) => new ShapedArray(a.shape.slice(1), a.dtype, a.weakType));
			builder.steps.push({
				type: "scan",
				plan: scanPlan,
				bodyProgram,
				bodyJaxpr,
				length,
				numCarry,
				numConsts,
				numX,
				numY,
				reverse,
				consts: constsIds,
				initCarry: initCarryIds,
				xs: xsIds,
				xsAvals: xsSliceAvals,
				outputs: outputIds$1
			});
			continue;
		}
		if (eqn.primitive === Primitive.DynamicUpdateSlice) throw new Error("DynamicUpdateSlice JIT compilation is not supported. DUS should only appear inside scan bodies, which handle it directly.");
		if (routinePrimitives.has(eqn.primitive)) {
			const routine = new require_backend.Routine(routinePrimitives.get(eqn.primitive), {
				inputShapes: eqn.inputs.map((x) => x.aval.shape),
				inputDtypes: eqn.inputs.map((x) => x.aval.dtype),
				outputShapes: eqn.outBinders.map((x) => x.aval.shape),
				outputDtypes: eqn.outBinders.map((x) => x.aval.dtype)
			}, eqn.params);
			const inputs = [];
			for (const input of eqn.inputs) if (input instanceof Var) {
				const jv = ctx.get(input);
				if (jv.type !== "imm") throw new Error(`jit: routine primitive ${eqn.primitive} input is not imm`);
				inputs.push(jv.arg);
			} else if (input instanceof Lit) inputs.push(builder.pushLit(input));
			const outputs = [];
			for (const outVar of eqn.outBinders) {
				const outId = builder.pushBuffer(outVar.aval.size * require_backend.byteWidth(outVar.aval.dtype));
				outputs.push(outId);
				ctx.set(outVar, {
					type: "imm",
					arg: outId
				});
			}
			builder.pushRoutine(routine, inputs, outputs);
			continue;
		}
		const inputExps = [];
		const inputAvals = [];
		const inputArgs = [];
		let inputReduction = null;
		const addArgs = (args) => {
			const newGids = [];
			for (const jitId of args) {
				let newGid = inputArgs.indexOf(jitId);
				if (newGid === -1) {
					newGid = inputArgs.length;
					inputArgs.push(jitId);
				}
				newGids.push(newGid);
			}
			return newGids;
		};
		for (const input of eqn.inputs) if (input instanceof Var) {
			const jv = ctx.get(input);
			if (jv.type === "exp") {
				const newGids = addArgs(jv.args);
				inputExps.push(jv.exp.reindexGids(newGids));
			} else if (jv.type === "imm") {
				const [gid] = addArgs([jv.arg]);
				const st = require_backend.ShapeTracker.fromShape(input.aval.shape);
				const indices = require_backend.unravelAlu(st.shape, require_backend.AluVar.gidx);
				inputExps.push(require_backend.AluExp.globalView(input.aval.dtype, gid, st, indices));
			} else if (jv.type === "red") {
				if (inputReduction) throw new Error("jit: unexpected, multiple red inputs");
				const newGids = addArgs(jv.args);
				inputExps.push(jv.reduction.epilogue.reindexGids(newGids));
				inputReduction = jv;
			}
			inputAvals.push(input.aval);
		} else if (input instanceof Lit) {
			inputExps.push(require_backend.AluExp.const(input.dtype, input.value));
			inputAvals.push(input.aval);
		} else throw new TypeError(`Unexpected input in Jaxpr: ${input}`);
		const rule = jitRules[eqn.primitive];
		if (!rule) throw new TypeError(`JIT not implemented for primitive ${eqn.primitive}`);
		let exp$2;
		let reduction;
		if (inputReduction) {
			const jv = inputReduction;
			const newEpilogue = rule(inputExps, inputAvals, eqn.params).exp[0];
			exp$2 = [jv.exp.reindexGids(addArgs(jv.args))];
			reduction = new require_backend.Reduction(jv.reduction.dtype, jv.reduction.op, jv.reduction.size, newEpilogue);
		} else {
			const ruleOutput = rule(inputExps, inputAvals, eqn.params);
			exp$2 = ruleOutput.exp;
			reduction = ruleOutput.reduction;
		}
		for (let i$1 = 0; i$1 < eqn.outBinders.length; i$1++) {
			const outVar = eqn.outBinders[i$1];
			if (blackNodes.has(outVar)) {
				const nargs$1 = inputArgs.length;
				const size$1 = outVar.aval.size;
				const kernel = new require_backend.Kernel(nargs$1, size$1, exp$2[i$1], reduction);
				const outId = builder.pushKernel(kernel, inputArgs);
				ctx.set(outVar, {
					type: "imm",
					arg: outId
				});
			} else if (reduction) ctx.set(outVar, {
				type: "red",
				exp: exp$2[i$1],
				reduction,
				args: inputArgs
			});
			else ctx.set(outVar, {
				type: "exp",
				exp: exp$2[i$1],
				args: inputArgs
			});
		}
	}
	const outputIds = [];
	for (const out of jaxpr.outs) if (out instanceof Var) {
		const jitValue = ctx.get(out);
		if (jitValue.type !== "imm") throw new Error("internal: Expected imm, since outs are black nodes");
		outputIds.push(jitValue.arg);
	} else if (out instanceof Lit) outputIds.push(builder.pushLit(out));
	const outputNeedsRef = new Set(require_backend.range(nargs));
	for (const outputId of outputIds) if (outputNeedsRef.has(outputId)) builder.pushIncref(outputId);
	else outputNeedsRef.add(outputId);
	builder.insertFreeSteps(outputIds);
	builder.recycleBuffers();
	const jp = new JitProgram(backend, builder.steps, require_backend.range(0, nargs), outputIds);
	if (require_backend.DEBUG >= 4) console.info(jp.toString());
	jitCompileCache.set(cacheKey, jp);
	return jp;
}
function reshapeViews(exp$2, mapping, reduceAxis = false) {
	return exp$2.rewrite((exp$3) => {
		if (exp$3.op === require_backend.AluOp.GlobalView) {
			const [gid, st] = exp$3.arg;
			const newSt = mapping(st);
			if (newSt) {
				const indices = reduceAxis ? require_backend.unravelAlu(newSt.shape.slice(0, -1), require_backend.AluVar.gidx).concat(require_backend.AluVar.ridx) : require_backend.unravelAlu(newSt.shape, require_backend.AluVar.gidx);
				return require_backend.AluExp.globalView(exp$3.dtype, gid, newSt, indices);
			}
		} else if (exp$3.op === require_backend.AluOp.GlobalIndex) throw new Error("internal: reshapeViews() called with GlobalIndex op");
	});
}
function broadcastedJit(fn, opts) {
	return (exps, avals, params) => {
		let { shape: newShape, dtype: newDtype } = avals.reduce(promoteAvals);
		const skipCastIdx = opts?.skipCastIdx ?? [];
		if (skipCastIdx.length) newDtype = avals.filter((_, i) => !skipCastIdx.includes(i)).reduce(promoteAvals).dtype;
		exps = exps.map((exp$2, i) => {
			exp$2 = reshapeViews(exp$2, (st) => {
				if (!require_backend.deepEqual(st.shape, newShape)) return st.broadcast(newShape, require_backend.range(newShape.length - st.shape.length));
			});
			if (exp$2.dtype !== newDtype && !skipCastIdx.includes(i)) exp$2 = require_backend.AluExp.cast(newDtype, exp$2);
			return exp$2;
		});
		return { exp: [fn(exps, params)] };
	};
}
function unopJit(fn) {
	return ([a], [_as], params) => {
		return { exp: [fn(a, params)] };
	};
}
function reshapeJit(fn) {
	return ([a], [_as], params) => {
		return { exp: [reshapeViews(a, (st) => fn(st, params))] };
	};
}
function routineNoJit() {
	return () => {
		throw new Error("jit: rule is not implemented for routines");
	};
}
const jitRules = {
	[Primitive.Add]: broadcastedJit(([a, b]) => require_backend.AluExp.add(a, b)),
	[Primitive.Mul]: broadcastedJit(([a, b]) => require_backend.AluExp.mul(a, b)),
	[Primitive.Idiv]: broadcastedJit(([a, b]) => require_backend.AluExp.idiv(a, b)),
	[Primitive.Mod]: broadcastedJit(([a, b]) => require_backend.AluExp.mod(a, b)),
	[Primitive.Min]: broadcastedJit(([a, b]) => require_backend.AluExp.min(a, b)),
	[Primitive.Max]: broadcastedJit(([a, b]) => require_backend.AluExp.max(a, b)),
	[Primitive.Neg]: unopJit((a) => require_backend.AluExp.sub(require_backend.AluExp.const(a.dtype, 0), a)),
	[Primitive.Reciprocal]: unopJit(require_backend.AluExp.reciprocal),
	[Primitive.Floor]: unopJit(require_backend.AluExp.floor),
	[Primitive.Ceil]: unopJit(require_backend.AluExp.ceil),
	[Primitive.StopGradient]: unopJit((a) => a),
	[Primitive.Cast]: unopJit((a, { dtype }) => require_backend.AluExp.cast(dtype, a)),
	[Primitive.Bitcast]: unopJit((a, { dtype }) => require_backend.AluExp.bitcast(dtype, a)),
	[Primitive.Sin]: unopJit(require_backend.AluExp.sin),
	[Primitive.Cos]: unopJit(require_backend.AluExp.cos),
	[Primitive.Asin]: unopJit(require_backend.AluExp.asin),
	[Primitive.Atan]: unopJit(require_backend.AluExp.atan),
	[Primitive.Exp]: unopJit(require_backend.AluExp.exp),
	[Primitive.Log]: unopJit(require_backend.AluExp.log),
	[Primitive.Erf]: unopJit(require_backend.AluExp.erf),
	[Primitive.Erfc]: unopJit(require_backend.AluExp.erfc),
	[Primitive.Sqrt]: unopJit(require_backend.AluExp.sqrt),
	[Primitive.Reduce]([a], [as], { op, axis }) {
		const keptAxes = [];
		const shiftedAxes = [];
		const newShape = [];
		for (let i = 0; i < as.shape.length; i++) if (axis.includes(i)) shiftedAxes.push(i);
		else {
			keptAxes.push(i);
			newShape.push(as.shape[i]);
		}
		const reductionSize = require_backend.prod(shiftedAxes.map((ax) => as.shape[ax]));
		newShape.push(reductionSize);
		const perm = keptAxes.concat(shiftedAxes);
		a = reshapeViews(a, (st) => st.permute(perm).reshape(newShape), true);
		const reduction = new require_backend.Reduction(a.dtype, op, reductionSize);
		return {
			exp: [a],
			reduction
		};
	},
	[Primitive.Pool]: reshapeJit((st, { window, strides }) => pool(st, window, strides)),
	[Primitive.PoolTranspose]([a], [as], { inShape, window, strides }) {
		let stX = poolTranspose(require_backend.ShapeTracker.fromShape(as.shape), inShape, window, strides);
		stX = stX.reshape([...inShape, require_backend.prod(stX.shape.slice(inShape.length))]);
		a = reshapeViews(a, (st) => st.compose(stX), true);
		const reduction = new require_backend.Reduction(a.dtype, require_backend.AluOp.Add, stX.shape[stX.shape.length - 1]);
		return {
			exp: [a],
			reduction
		};
	},
	[Primitive.Dot]([a, b], [as, bs]) {
		const k1 = jitRules[Primitive.Mul]([a, b], [as, bs], {});
		const [c] = k1.exp;
		const cs = promoteAvals(as, bs);
		return jitRules[Primitive.Reduce]([c], [cs], {
			op: require_backend.AluOp.Add,
			axis: [cs.ndim - 1]
		});
	},
	[Primitive.Conv]([a, b], [as, bs], params) {
		const [stX, stY] = prepareConv(require_backend.ShapeTracker.fromShape(as.shape), require_backend.ShapeTracker.fromShape(bs.shape), params);
		a = reshapeViews(a, (st) => st.compose(stX));
		b = reshapeViews(b, (st) => st.compose(stY));
		as = new ShapedArray(stX.shape, as.dtype, as.weakType);
		bs = new ShapedArray(stY.shape, bs.dtype, bs.weakType);
		return jitRules[Primitive.Dot]([a, b], [as, bs], {});
	},
	[Primitive.Compare]: broadcastedJit(([a, b], { op }) => aluCompare(a, b, op)),
	[Primitive.Where]: broadcastedJit(([cond, a, b]) => require_backend.AluExp.where(cond, a, b), { skipCastIdx: [0] }),
	[Primitive.Concatenate](exps, avals, { axis }) {
		const ndim$2 = avals[0].ndim;
		const sizes = avals.map((x) => x.shape[axis]);
		const finalSize = sizes.reduce((a, b) => a + b, 0);
		const { dtype: dtypeOut } = avals.map((x) => x.scalar()).reduce(promoteAvals);
		const makePadAxis = (start, end) => require_backend.range(ndim$2).map((i) => i === axis ? [start, end] : [0, 0]);
		let cum = 0;
		const src = [];
		for (let i = 0; i < exps.length; i++) {
			const padding = makePadAxis(cum, finalSize - cum - sizes[i]);
			src.push(reshapeViews(require_backend.AluExp.cast(dtypeOut, exps[i]), (st) => st.pad(padding)));
			cum += sizes[i];
		}
		return { exp: [src.reduce(require_backend.AluExp.add)] };
	},
	[Primitive.Split]([a], [as], { axis, sizes }) {
		const exp$2 = [];
		let start = 0;
		for (const size$1 of sizes) {
			const slice = require_backend.range(as.ndim).map((d) => d === axis ? [start, start + size$1] : [0, as.shape[d]]);
			exp$2.push(reshapeViews(a, (st) => st.shrink(slice)));
			start += size$1;
		}
		return { exp: exp$2 };
	},
	[Primitive.RandomBits]: (keys, keyShapes, { shape: shape$1, mode }) => {
		const keyShape = keyShapes[0].shape;
		const mapping = (st) => {
			if (!require_backend.deepEqual(st.shape, shape$1)) return st.broadcast(shape$1, require_backend.range(st.shape.length, shape$1.length));
		};
		const k0 = reshapeViews(keys[0], mapping);
		const k1 = reshapeViews(keys[1], mapping);
		const c0 = require_backend.AluExp.u32(0);
		const c1 = require_backend.AluExp.mod(require_backend.AluExp.cast(require_backend.DType.Uint32, require_backend.AluVar.gidx), require_backend.AluExp.u32(Math.max(require_backend.prod(shape$1.slice(keyShape.length)), 1)));
		const exp$2 = require_backend.AluExp.threefry2x32(k0, k1, c0, c1, mode);
		return { exp: [exp$2] };
	},
	[Primitive.Gather]([x, ...indices], [xs, ...indicesShapes], { axis, outDim }) {
		const axisSet = new Set(axis);
		const indexShape = indicesShapes.map((c) => c.shape).reduce(require_backend.generalBroadcast);
		const finalShape = xs.shape.filter((_, i) => !axisSet.has(i));
		finalShape.splice(outDim, 0, ...indexShape);
		const idxAll = require_backend.unravelAlu(finalShape, require_backend.AluVar.gidx);
		const idxNonaxis = [...idxAll];
		idxNonaxis.splice(outDim, indexShape.length);
		const src = [...idxNonaxis];
		for (let i = 0; i < xs.shape.length; i++) if (axisSet.has(i)) src.splice(i, 0, null);
		for (const [i, iexp] of indices.entries()) src[axis[i]] = require_backend.AluExp.cast(require_backend.DType.Int32, reshapeViews(iexp, (st) => st.broadcast(finalShape, [...require_backend.range(outDim + indexShape.length - st.shape.length), ...require_backend.range(outDim + indexShape.length, finalShape.length)])));
		const [index, valid] = require_backend.ShapeTracker.fromShape(xs.shape).toAluExp(src);
		if (!valid.resolve()) throw new Error("internal: expected full validity mask in Gather");
		return { exp: [x.substitute({ gidx: index })] };
	},
	[Primitive.Transpose]: reshapeJit((st, { perm }) => st.permute(perm)),
	[Primitive.Broadcast]: reshapeJit((st, { shape: shape$1, axis }) => st.broadcast(shape$1, axis)),
	[Primitive.Reshape]: reshapeJit((st, { shape: shape$1 }) => st.reshape(shape$1)),
	[Primitive.Flip]: reshapeJit((st, { axis }) => {
		const arg = require_backend.rep(st.shape.length, false);
		for (const ax of axis) arg[ax] = true;
		return st.flip(arg);
	}),
	[Primitive.Shrink]: reshapeJit((st, { slice }) => st.shrink(slice)),
	[Primitive.Pad]: reshapeJit((st, { width }) => st.pad(width)),
	[Primitive.Sort]: routineNoJit(),
	[Primitive.Argsort]: routineNoJit(),
	[Primitive.TriangularSolve]: routineNoJit(),
	[Primitive.Cholesky]: routineNoJit(),
	[Primitive.LU]: routineNoJit(),
	[Primitive.Jit]() {
		throw new Error("internal: Jit should have been flattened before JIT compilation");
	},
	[Primitive.DynamicUpdateSlice]() {
		throw new Error("internal: DynamicUpdateSlice is handled specially in jitCompile");
	},
	[Primitive.Scan]() {
		throw new Error("internal: Scan is handled specially in jitCompile, not via jitRules");
	}
};
/** Determines how to split the Jaxpr into kernels via dataflow analysis. */
function splitGraphDataflow(backend, jaxpr) {
	const varToDefn = /* @__PURE__ */ new Map();
	const varToUsages = /* @__PURE__ */ new Map();
	for (let i = 0; i < jaxpr.eqns.length; i++) {
		const eqn = jaxpr.eqns[i];
		for (const v of eqn.outBinders) if (v instanceof Var) varToDefn.set(v, i);
		for (const input of eqn.inputs) if (input instanceof Var) {
			const usages = varToUsages.get(input);
			if (usages) usages.push(i);
			else varToUsages.set(input, [i]);
		}
	}
	const reducePrimitives = [
		Primitive.Reduce,
		Primitive.Dot,
		Primitive.Conv,
		Primitive.PoolTranspose
	];
	const reductionEpilogueEqns = /* @__PURE__ */ new Set();
	const reductionEndpointEqns = /* @__PURE__ */ new Set();
	for (let i = 0; i < jaxpr.eqns.length; i++) {
		const eqn = jaxpr.eqns[i];
		if (reducePrimitives.includes(eqn.primitive)) {
			let head = i;
			while (true) {
				reductionEpilogueEqns.add(head);
				const outVar = jaxpr.eqns[head].outBinders[0];
				const usages = varToUsages.get(outVar) ?? [];
				if (jaxpr.outs.includes(outVar) || usages.length !== 1) break;
				if (reductionEpilogueEqns.has(usages[0])) break;
				const nextEqn = jaxpr.eqns[usages[0]];
				switch (nextEqn.primitive) {
					case Primitive.Neg:
					case Primitive.Reciprocal:
					case Primitive.Floor:
					case Primitive.Ceil:
					case Primitive.StopGradient:
					case Primitive.Cast:
					case Primitive.Bitcast:
					case Primitive.Sin:
					case Primitive.Cos:
					case Primitive.Asin:
					case Primitive.Atan:
					case Primitive.Exp:
					case Primitive.Log:
					case Primitive.Erf:
					case Primitive.Erfc:
					case Primitive.Sqrt:
						head = usages[0];
						continue;
					case Primitive.Add:
					case Primitive.Mul:
					case Primitive.Idiv:
					case Primitive.Mod:
					case Primitive.Min:
					case Primitive.Max: {
						const otherInput = nextEqn.inputs.find((v) => v !== outVar);
						if (otherInput instanceof Lit || require_backend.deepEqual(require_backend.generalBroadcast(otherInput.aval.shape, outVar.aval.shape), outVar.aval.shape)) {
							head = usages[0];
							continue;
						}
						break;
					}
				}
				break;
			}
			reductionEndpointEqns.add(head);
		}
	}
	const blackNodes = /* @__PURE__ */ new Set();
	const p1NextBlack = /* @__PURE__ */ new Map();
	for (const v of jaxpr.outs) if (v instanceof Var) {
		blackNodes.add(v);
		p1NextBlack.set(v, v);
	}
	const heterogeneousViewPrimitives = [Primitive.RandomBits, Primitive.Gather];
	const needsCleanShapePrimitives = [Primitive.Concatenate, Primitive.Pad];
	const specialBlackPrimitives = [Primitive.Scan, Primitive.DynamicUpdateSlice];
	for (let i = jaxpr.eqns.length - 1; i >= 0; i--) {
		const eqn = jaxpr.eqns[i];
		if (reductionEndpointEqns.has(i) || heterogeneousViewPrimitives.includes(eqn.primitive) || routinePrimitives.has(eqn.primitive) || specialBlackPrimitives.includes(eqn.primitive) || eqn.outBinders.some((v) => blackNodes.has(v))) {
			for (const v of eqn.outBinders) {
				blackNodes.add(v);
				p1NextBlack.set(v, v);
			}
			continue;
		}
		const reach = /* @__PURE__ */ new Set();
		let needsCleanOutput = false;
		outer: for (const v of eqn.outBinders) for (const j of varToUsages.get(v) ?? []) {
			if (needsCleanShapePrimitives.includes(jaxpr.eqns[j].primitive) || routinePrimitives.has(jaxpr.eqns[j].primitive) || specialBlackPrimitives.includes(jaxpr.eqns[j].primitive)) {
				needsCleanOutput = true;
				break outer;
			}
			for (const o of jaxpr.eqns[j].outBinders) {
				const u = p1NextBlack.get(o);
				if (u) reach.add(u);
			}
		}
		if (reach.size > 1 || needsCleanOutput) for (const v of eqn.outBinders) {
			blackNodes.add(v);
			p1NextBlack.set(v, v);
		}
		else if (reach.size === 1) {
			const b = reach.values().next().value;
			for (const v of eqn.outBinders) p1NextBlack.set(v, b);
		}
	}
	const p2Deps = /* @__PURE__ */ new Map();
	for (const v of jaxpr.inBinders) p2Deps.set(v, new Set([v]));
	let p2idx = 0;
	while (p2idx < jaxpr.eqns.length) {
		const eqn = jaxpr.eqns[p2idx++];
		const deps = [];
		for (const input of eqn.inputs) if (input instanceof Var) if (blackNodes.has(input)) deps.push(new Set([input]));
		else deps.push(p2Deps.get(input));
		else deps.push(/* @__PURE__ */ new Set());
		const depCounter = /* @__PURE__ */ new Map();
		for (const depSet of deps) for (const dep of depSet) depCounter.set(dep, (depCounter.get(dep) ?? 0) + 1);
		if (depCounter.size > backend.maxArgs) {
			let maxUniqueDeps = 0;
			let assocInput = -1;
			for (let i = 0; i < eqn.inputs.length; i++) {
				const input = eqn.inputs[i];
				if (input instanceof Var && varToDefn.has(input)) {
					let uniqueDeps = 0;
					for (const dep of deps[i]) if (depCounter.get(dep) === 1) uniqueDeps++;
					if (uniqueDeps > maxUniqueDeps) {
						maxUniqueDeps = uniqueDeps;
						assocInput = i;
					}
				}
			}
			if (assocInput === -1) throw new Error(`internal: maxArgs, no input found to mark as black in Jaxpr equation ${eqn}`);
			const assocVar = eqn.inputs[assocInput];
			p2idx = varToDefn.get(assocVar);
			for (const out of jaxpr.eqns[p2idx++].outBinders) blackNodes.add(out);
		} else {
			const s = new Set(depCounter.keys());
			for (const out of eqn.outBinders) p2Deps.set(out, s);
		}
	}
	return blackNodes;
}

//#endregion
//#region src/frontend/array.ts
var import_usingCtx$10 = /* @__PURE__ */ __toESM(require_usingCtx(), 1);
const JsArray$1 = globalThis.Array;
const inlineArrayLimit = 128;
/** Version of pureArray with fudged types. */
const fudgeArray = pureArray;
/**
* An executable operation that will be dispatched to the backend.
*
* This holds a reference to all input buffers used in the operation. After the
* operation is dispatched, the references should be released.
*/
var PendingExecute = class {
	prepared = null;
	submitted = false;
	#promise = null;
	#rc = 1;
	constructor(backend, source, inputs, outputs) {
		this.backend = backend;
		this.source = source;
		this.inputs = inputs;
		this.outputs = outputs;
		for (const slot of inputs) this.backend.incRef(slot);
		for (const slot of outputs) this.backend.incRef(slot);
	}
	updateRc(delta) {
		if (this.#rc <= 0) throw new Error("internal: PendingExecute used rc<=0");
		this.#rc += delta;
		if (this.#rc <= 0 && !this.submitted) {
			for (const slot of this.inputs) this.backend.decRef(slot);
			for (const slot of this.outputs) this.backend.decRef(slot);
		}
	}
	async prepare() {
		if (this.prepared) return;
		if (this.#promise) {
			await this.#promise;
			return;
		}
		this.#promise = (async () => {
			if (this.source instanceof require_backend.Kernel) this.prepared = await this.backend.prepareKernel(this.source);
			else this.prepared = await this.backend.prepareRoutine(this.source);
		})();
		await this.#promise;
	}
	prepareSync() {
		if (this.prepared) return;
		if (this.source instanceof require_backend.Kernel) this.prepared = this.backend.prepareKernelSync(this.source);
		else this.prepared = this.backend.prepareRoutineSync(this.source);
	}
	submit() {
		if (this.submitted) return;
		if (this.#rc <= 0) throw new Error("internal: PendingExecute used rc<=0");
		if (!this.prepared) throw new Error("internal: Not prepared yet");
		this.submitted = true;
		this.backend.dispatch(this.prepared, this.inputs, this.outputs);
		for (const slot of this.inputs) this.backend.decRef(slot);
		for (const slot of this.outputs) this.backend.decRef(slot);
	}
};
/**
* A multidimensional numeric array with data stored on CPU or GPU.
*
* This is the library's core data type. Equivalent to `jax.Array` from JAX, or
* `torch.Tensor`.
*
* Not to be confused with the JavaScript "Array" constructor. Avoid importing
* this into your code's namespace if you're already using the JavaScript
* "Array" type by name.
*/
var Array$1 = class Array$1 extends Tracer {
	#dtype;
	#weakType;
	#source;
	#st;
	#backend;
	#committed;
	#rc;
	#pendingSet;
	/**
	* @ignore
	* Constructs an array from source, shape and backend. Note that if the source
	* is a backend `Slot`, this constructor _takes ownership_ of the slot. It
	* will be freed when the array is disposed.
	*/
	constructor(args) {
		super(baseArrayTrace);
		this.#dtype = args.dtype;
		this.#weakType = args.weakType;
		this.#source = args.source;
		this.#st = args.st;
		this.#backend = args.backend;
		this.#committed = args.committed;
		this.#rc = 1;
		this.#pendingSet = new Set(args.pending);
		if (this.#pendingSet.size === 0) this.#pendingSet = null;
		else if (this.#source instanceof require_backend.AluExp) throw new Error("internal: AluExp source cannot have pending executes");
		if (_leakTrackingEnabled) _leakTrackingMap.set(this, /* @__PURE__ */ new Error());
		if (_peArrayCreationTracker) _peArrayCreationTracker.push(this);
	}
	/** @ignore */
	get aval() {
		return new ShapedArray(this.#st.shape, this.#dtype, this.#weakType);
	}
	/** Return a simple string representation of the array's dimensions. */
	toString() {
		return `Array:${this.#dtype}[${this.shape.join(",")}]`;
	}
	get device() {
		return this.#backend.type;
	}
	#check() {
		if (this.#rc <= 0) throw new UseAfterFreeError(this);
	}
	/** Construct an array, copying fields from `this`. */
	#newArrayFrom(args) {
		return new Array$1({
			source: args.source ?? this.#source,
			st: args.st ?? this.#st,
			dtype: args.dtype ?? this.#dtype,
			weakType: this.#weakType,
			backend: args.backend ?? this.#backend,
			committed: args.committed ?? this.#committed,
			pending: args.pending ?? this.#pending ?? void 0
		});
	}
	get ref() {
		this.#check();
		this.#rc++;
		if (_trackRefsEnabled) _lastRefMap.set(this, /* @__PURE__ */ new Error());
		return this;
	}
	/** Get the current reference count (for debugging memory management). */
	get refCount() {
		return this.#rc;
	}
	dispose() {
		this.#check();
		if (--this.#rc === 0) {
			if (_leakTrackingMap.size > 0) _leakTrackingMap.delete(this);
			for (const exe of this.#pending) exe.updateRc(-1);
			if (typeof this.#source === "number") this.#backend.decRef(this.#source);
		}
	}
	/**
	* Override Tracer's no-op to actually dispose concrete Arrays.
	*
	* During PE tracing, concrete Arrays are created by PE's all-known evaluation
	* path and tracked in knownIntermediates. Their lifecycle is managed by
	* disposePeIntermediates after PE completes. If `using` declarations in JVP
	* helpers also dispose these arrays, the double-dispose causes
	* UseAfterFreeError when the ClosedJaxpr later accesses its consts.
	*
	* We detect PE scope via _peArrayCreationTracker (non-null when inside PE).
	* JVP-only traces (no PE) still dispose normally — those intermediates have
	* real Slots that need cleanup.
	*/
	[Symbol.dispose]() {
		if (!_peArrayCreationTracker && this.#rc > 0) this.dispose();
	}
	/** Get the pending executes as a list, trimming if already submitted. */
	get #pending() {
		if (!this.#pendingSet) return [];
		for (const p of this.#pendingSet) if (p.submitted) this.#pendingSet.delete(p);
		if (this.#pendingSet.size === 0) {
			this.#pendingSet = null;
			return [];
		} else return [...this.#pendingSet];
	}
	/**
	* Convert this array into a primitive value.
	*
	* This only works for scalars (0-dimensional arrays). It lets you get values
	* "out" of the JAX system. For instance, if `x = np.array(5)`, then you can
	* evaluate `x + 1` and `x ** 2` to get `6` and `25`, respectively.
	*
	* This method is also called for `==` equality.
	*/
	[Symbol.toPrimitive]() {
		if (this.ndim === 0) return this.dataSync()[0];
		else throw new Error(`Cannot convert non-scalar array to primitive: ${this.toString()}`);
	}
	#reshape(st) {
		this.#check();
		const pending = this.#pending;
		for (const exe of pending) exe.updateRc(1);
		if (typeof this.#source === "number") this.#backend.incRef(this.#source);
		return this.#newArrayFrom({
			st,
			pending
		});
	}
	/**
	* Underlying implementation of the Gather primitive. This indexes an array
	* and extracts slices based on indices in other integer arrays.
	*/
	#gather(indices, axis, outDim) {
		this.#check();
		const axisSet = new Set(axis);
		if (axisSet.size !== axis.length) throw new TypeError("Gather axis must not have duplicates");
		if (indices.some((a) => a.#committed && a.#backend !== this.#backend)) throw new TypeError(`Gather indices must have the same backend: ${this.#backend.type}`);
		indices = indices.map((ar) => ar._putSync(this.#backend));
		indices = Array$1.#broadcastArrays(indices);
		const indexShape = indices[0].shape;
		const finalShape = this.shape.filter((_, i) => !axisSet.has(i));
		finalShape.splice(outDim, 0, ...indexShape);
		const idxAll = require_backend.unravelAlu(finalShape, require_backend.AluVar.gidx);
		const idxNonaxis = [...idxAll];
		const idxAxis = idxNonaxis.splice(outDim, indexShape.length);
		const inputs = [];
		const src = [...idxNonaxis];
		for (let i = 0; i < this.shape.length; i++) if (axisSet.has(i)) src.splice(i, 0, null);
		for (const [i, ar] of indices.entries()) if (ar.#source instanceof require_backend.AluExp) src[axis[i]] = require_backend.AluExp.cast(require_backend.DType.Int32, require_backend.accessorAluExp(ar.#source, ar.#st, idxAxis));
		else {
			let gid = inputs.indexOf(ar.#source);
			if (gid === -1) {
				gid = inputs.length;
				inputs.push(ar.#source);
			}
			src[axis[i]] = require_backend.AluExp.cast(require_backend.DType.Int32, require_backend.AluExp.globalView(ar.#dtype, gid, ar.#st, idxAxis));
		}
		let exp$2;
		if (this.#source instanceof require_backend.AluExp) exp$2 = require_backend.accessorAluExp(this.#source, this.#st, src);
		else {
			let gid = inputs.indexOf(this.#source);
			if (gid === -1) {
				gid = inputs.length;
				inputs.push(this.#source);
			}
			exp$2 = require_backend.accessorGlobal(this.#dtype, gid, this.#st, src);
		}
		const kernel = new require_backend.Kernel(inputs.length, require_backend.prod(finalShape), exp$2);
		const output = this.#backend.malloc(kernel.bytes);
		const pending = [...this.#pending, ...indices.flatMap((ar) => ar.#pending)];
		for (const exe of pending) exe.updateRc(1);
		pending.push(new PendingExecute(this.#backend, kernel, inputs, [output]));
		return this.#newArrayFrom({
			source: output,
			st: require_backend.ShapeTracker.fromShape(finalShape),
			pending
		});
	}
	/** Move axes to the rightmost dimension of the shape. */
	#moveAxesDown(axis) {
		this.#check();
		if (axis.length === 0) return this.reshape(this.shape.concat(1));
		const newShape = [];
		const keptAxes = [];
		const shiftedAxes = [];
		for (let i = 0; i < this.#st.shape.length; i++) if (axis.includes(i)) shiftedAxes.push(i);
		else {
			keptAxes.push(i);
			newShape.push(this.#st.shape[i]);
		}
		newShape.push(-1);
		const transposed = this.#transpose(keptAxes.concat(shiftedAxes));
		const result = transposed.reshape(newShape);
		transposed.dispose();
		return result;
	}
	#transpose(perm) {
		this.#check();
		if (!require_backend.isPermutation(perm, this.ndim)) throw new Error(`Invalid perm for transpose: ${JSON.stringify(perm)}`);
		return this.#reshape(this.#st.permute(perm));
	}
	#unary(op, dtypeOutput) {
		const weakType = !dtypeOutput && this.#weakType;
		dtypeOutput ??= this.#dtype;
		this.#check();
		if (this.#source instanceof require_backend.AluExp) {
			const exp$3 = new require_backend.AluExp(op, dtypeOutput, [this.#source]);
			return this.#newArrayFrom({
				source: exp$3.simplify(),
				dtype: dtypeOutput,
				weakType
			});
		}
		const indices = require_backend.unravelAlu(this.#st.shape, require_backend.AluVar.gidx);
		const exp$2 = new require_backend.AluExp(op, dtypeOutput, [require_backend.AluExp.globalView(this.#dtype, 0, this.#st, indices)]);
		const kernel = new require_backend.Kernel(1, this.#st.size, exp$2);
		const output = this.#backend.malloc(kernel.bytes);
		const pending = [...this.#pending];
		for (const exe of pending) exe.updateRc(1);
		pending.push(new PendingExecute(this.#backend, kernel, [this.#source], [output]));
		return this.#newArrayFrom({
			source: output,
			st: require_backend.ShapeTracker.fromShape(this.shape),
			dtype: dtypeOutput,
			weakType,
			pending
		});
	}
	#binary(op, other) {
		const custom = (src) => new require_backend.AluExp(op, src[0].dtype, src);
		return Array$1.#naryCustom(op, custom, [this, other]);
	}
	static #naryCustom(name, custom, arrays, { dtypeOverride, strongTypeOutput, reduceAxis } = {}) {
		const n = arrays.length;
		if (n === 0) throw new TypeError(`No inputs for ${name}`);
		for (const ar of arrays) ar.#check();
		let castDtype;
		let castWeakType = true;
		for (let i = 0; i < n; i++) if (dtypeOverride?.[i]) {
			if (arrays[i].#dtype !== dtypeOverride[i]) throw new TypeError(`Wrong dtype in ${name}: expected ${dtypeOverride[i]}, got ${arrays[i].#dtype}`);
		} else if (castDtype === void 0) {
			castDtype = arrays[i].#dtype;
			castWeakType = arrays[i].#weakType;
		} else ({dtype: castDtype, weakType: castWeakType} = promoteAvals(new ShapedArray([], castDtype, castWeakType), arrays[i].aval.scalar()));
		const weakType = castWeakType && !strongTypeOutput;
		const { backend, committed } = Array$1.#computeBackend(name, arrays);
		const originalArrays = [...arrays];
		arrays = arrays.map((ar) => ar._putSync(backend));
		const preBroadcast = [...arrays];
		arrays = Array$1.#broadcastArrays(arrays);
		const newShape = [...arrays[0].shape];
		if (arrays.every((ar) => ar.#source instanceof require_backend.AluExp) && !reduceAxis) {
			const sources = arrays.map((ar, i) => {
				if (!dtypeOverride?.[i]) return require_backend.AluExp.cast(castDtype, ar.#source);
				else return ar.#source;
			});
			if (arrays.every((ar) => require_backend.deepEqual(ar.#st, arrays[0].#st))) {
				const exp$4 = custom(sources);
				for (let i = 0; i < arrays.length; i++) if (arrays[i] !== preBroadcast[i]) arrays[i].dispose();
				for (let i = 0; i < originalArrays.length; i++) if (preBroadcast[i] !== originalArrays[i]) preBroadcast[i].dispose();
				return new Array$1({
					source: exp$4.simplify(),
					st: arrays[0].#st,
					dtype: exp$4.dtype,
					weakType,
					backend,
					committed
				});
			}
			const exp$3 = custom(arrays.map((ar, i) => {
				const src$1 = sources[i];
				if (ar.#st.contiguous) return src$1;
				return require_backend.accessorAluExp(src$1, ar.#st, require_backend.unravelAlu(newShape, require_backend.AluVar.idx));
			}));
			const st = require_backend.ShapeTracker.fromShape(newShape);
			for (let i = 0; i < arrays.length; i++) if (arrays[i] !== preBroadcast[i]) arrays[i].dispose();
			for (let i = 0; i < originalArrays.length; i++) if (preBroadcast[i] !== originalArrays[i]) preBroadcast[i].dispose();
			return new Array$1({
				source: exp$3.simplify(),
				st,
				dtype: exp$3.dtype,
				weakType,
				backend,
				committed
			});
		}
		let indices;
		if (!reduceAxis) indices = require_backend.unravelAlu(newShape, require_backend.AluVar.gidx);
		else {
			const contractedShape = newShape.slice(0, -1);
			indices = [...require_backend.unravelAlu(contractedShape, require_backend.AluVar.gidx), require_backend.AluVar.ridx];
		}
		const inputs = [];
		const src = [];
		for (const [i, ar] of arrays.entries()) {
			let nextSrc;
			if (ar.#source instanceof require_backend.AluExp) nextSrc = require_backend.accessorAluExp(ar.#source, ar.#st, indices);
			else {
				let gid = inputs.indexOf(ar.#source);
				if (gid === -1) {
					gid = inputs.length;
					inputs.push(ar.#source);
				}
				nextSrc = require_backend.AluExp.globalView(ar.#dtype, gid, ar.#st, indices);
			}
			if (!dtypeOverride?.[i]) nextSrc = require_backend.AluExp.cast(castDtype, nextSrc);
			src.push(nextSrc);
		}
		const exp$2 = custom(src);
		let re = void 0;
		if (reduceAxis) {
			const [axisSize] = newShape.splice(-1, 1);
			re = new require_backend.Reduction(exp$2.dtype, require_backend.AluOp.Add, axisSize);
		}
		const kernel = new require_backend.Kernel(inputs.length, require_backend.prod(newShape), exp$2, re);
		const output = backend.malloc(kernel.bytes);
		const pending = new Set([...arrays.flatMap((ar) => ar.#pending)]);
		for (const exe of pending) exe.updateRc(1);
		pending.add(new PendingExecute(backend, kernel, inputs, [output]));
		for (let i = 0; i < arrays.length; i++) if (arrays[i] !== preBroadcast[i]) arrays[i].dispose();
		for (let i = 0; i < originalArrays.length; i++) if (preBroadcast[i] !== originalArrays[i]) preBroadcast[i].dispose();
		return new Array$1({
			source: output,
			st: require_backend.ShapeTracker.fromShape(newShape),
			dtype: kernel.dtype,
			weakType,
			backend,
			committed,
			pending
		});
	}
	/** Reduce the last dimension of the array by an operation. */
	#reduce(op) {
		const shape$1 = this.shape;
		const reduction = new require_backend.Reduction(this.#dtype, op, shape$1[shape$1.length - 1]);
		const newShape = shape$1.slice(0, -1);
		const newSize = require_backend.prod(newShape);
		const indices = [...require_backend.unravelAlu(newShape, require_backend.AluVar.gidx), require_backend.AluVar.ridx];
		let exp$2;
		const inputs = [];
		if (this.#source instanceof require_backend.AluExp) exp$2 = require_backend.accessorAluExp(this.#source, this.#st, indices);
		else {
			inputs.push(this.#source);
			exp$2 = require_backend.accessorGlobal(this.#dtype, 0, this.#st, indices);
		}
		const kernel = new require_backend.Kernel(inputs.length, newSize, exp$2, reduction);
		const output = this.#backend.malloc(kernel.bytes);
		const pending = [...this.#pending];
		for (const exe of pending) exe.updateRc(1);
		pending.push(new PendingExecute(this.#backend, kernel, inputs, [output]));
		return this.#newArrayFrom({
			source: output,
			st: require_backend.ShapeTracker.fromShape(newShape),
			pending
		});
	}
	/** Apply an operation with custom lowering to this array. */
	static #routine(prim) {
		return (arrays, params) => {
			const { backend, committed } = Array$1.#computeBackend(prim, arrays);
			for (const ar of arrays) ar.#realize();
			const avals = arrays.map((ar) => ar.aval);
			const avalsOut = abstractEvalRules[prim](avals, params);
			const routine = new require_backend.Routine(routinePrimitives.get(prim), {
				inputShapes: avals.map((a) => a.shape),
				inputDtypes: avals.map((a) => a.dtype),
				outputShapes: avalsOut.map((a) => a.shape),
				outputDtypes: avalsOut.map((a) => a.dtype)
			}, params);
			const inputs = arrays.map((ar) => ar.#source);
			const outputs = avalsOut.map((x) => backend.malloc(require_backend.byteWidth(x.dtype) * x.size));
			const pending = arrays.flatMap((ar) => ar.#pending);
			for (const exe of pending) exe.updateRc(+outputs.length);
			pending.push(new PendingExecute(backend, routine, inputs, outputs));
			pending[pending.length - 1].updateRc(+outputs.length - 1);
			return outputs.map((output, i) => new Array$1({
				source: output,
				st: require_backend.ShapeTracker.fromShape(avalsOut[i].shape),
				dtype: avalsOut[i].dtype,
				weakType: avalsOut[i].weakType,
				backend,
				committed,
				pending
			}));
		};
	}
	/**
	* Normalizes this array into one backed by a `Slot`.
	*
	* This mutates the array in-place, turning it into an equivalent array whose
	* source is actual, contiguous data on device.
	*
	* Calling this twice is a no-op.
	*/
	#realize() {
		this.#check();
		const indices = require_backend.unravelAlu(this.#st.shape, require_backend.AluVar.gidx);
		if (this.#source instanceof require_backend.AluExp) {
			const exp$2 = require_backend.accessorAluExp(this.#source, this.#st, indices);
			const kernel = new require_backend.Kernel(0, this.#st.size, exp$2);
			const output = this.#backend.malloc(kernel.bytes);
			const pendingItem = new PendingExecute(this.#backend, kernel, [], [output]);
			this.#source = output;
			this.#st = require_backend.ShapeTracker.fromShape(this.shape);
			this.#pendingSet = new Set([pendingItem]);
		} else {
			if (this.#st.contiguous) return;
			const exp$2 = require_backend.accessorGlobal(this.#dtype, 0, this.#st, indices);
			const kernel = new require_backend.Kernel(1, this.#st.size, exp$2);
			const output = this.#backend.malloc(kernel.bytes);
			const pendingItem = new PendingExecute(this.#backend, kernel, [this.#source], [output]);
			this.#backend.decRef(this.#source);
			this.#source = output;
			this.#st = require_backend.ShapeTracker.fromShape(this.shape);
			this.#pendingSet ??= /* @__PURE__ */ new Set();
			this.#pendingSet.add(pendingItem);
		}
	}
	#dataInline() {
		try {
			var _usingCtx$1 = (0, import_usingCtx$10.default)();
			this.#check();
			if (!(this.#source instanceof require_backend.AluExp)) throw new Error("internal: #dataInline called on non-AluExp source");
			const ar = _usingCtx$1.u(this.#newArrayFrom({ backend: require_backend.getBackend("cpu") }));
			return ar.dataSync();
		} catch (_) {
			_usingCtx$1.e = _;
		} finally {
			_usingCtx$1.d();
		}
	}
	static #broadcastArrays(arrays) {
		if (arrays.length === 0) throw new Error("Need at least one array to broadcast");
		if (arrays.length === 1) return arrays;
		const newShape = arrays.map((a) => a.shape).reduce(require_backend.generalBroadcast);
		return arrays.map((ar) => {
			if (require_backend.deepEqual(ar.shape, newShape)) return ar;
			return ar.#reshape(ar.#st.broadcast(newShape, require_backend.range(newShape.length - ar.ndim)));
		});
	}
	static #computeBackend(name, arrays) {
		const committed = arrays.filter((ar) => ar.#committed);
		if (committed.length > 0) {
			const backend = committed[0].#backend;
			for (const ar of committed) if (ar.#backend !== backend) throw new Error(`Device mismatch in ${name} between committed arrays on (${backend.type}, ${ar.#backend.type}), please move to the same device with devicePut()`);
			return {
				backend,
				committed: true
			};
		} else {
			const backend = arrays.length > 0 ? arrays[0].#backend : require_backend.getBackend();
			return {
				backend,
				committed: false
			};
		}
	}
	/** Realize the array and return it as data. */
	async data() {
		if (this.#source instanceof require_backend.AluExp && this.size < inlineArrayLimit && this.device !== "cpu") return this.#dataInline();
		this.#realize();
		const pending = this.#pending;
		if (pending) {
			await Promise.all(pending.map((p) => p.prepare()));
			for (const p of pending) p.submit();
		}
		const byteCount = require_backend.byteWidth(this.#dtype) * this.size;
		const buf = await this.#backend.read(this.#source, 0, byteCount);
		return require_backend.dtypedArray(this.dtype, buf);
	}
	/**
	* Wait for this array to finish evaluation.
	*
	* Operations and data loading in jax-js are lazy, so this function ensures
	* that pending operations are dispatched and fully executed before it
	* returns.
	*
	* If you are mapping from `data()` or `dataSync()`, it will also trigger
	* dispatch of operations as well.
	*
	* **Note:** `jax.blockUntilReady()` is a higher-level API, it calls this
	* asynchronously for multiple arrays.
	*/
	async blockUntilReady() {
		this.#check();
		if (this.#source instanceof require_backend.AluExp) return this;
		const pending = this.#pending;
		if (pending) {
			await Promise.all(pending.map((p) => p.prepare()));
			for (const p of pending) p.submit();
		}
		await this.#backend.read(this.#source, 0, 0);
		return this;
	}
	/**
	* Realize the array and return it as data. This is a sync variant and not
	* recommended for performance reasons, as it will block rendering.
	*/
	dataSync() {
		if (this.#source instanceof require_backend.AluExp && this.size < inlineArrayLimit && this.device !== "cpu") return this.#dataInline();
		this.#realize();
		for (const p of this.#pending) {
			p.prepareSync();
			p.submit();
		}
		const byteCount = require_backend.byteWidth(this.#dtype) * this.size;
		const buf = this.#backend.readSync(this.#source, 0, byteCount);
		return require_backend.dtypedArray(this.dtype, buf);
	}
	/**
	* Convert this array into a JavaScript object.
	*
	* This is a blocking operation that will compile all of the shaders and wait
	* for execution to complete, synchronously. No other JavaScript code on the
	* site will be run during shader execution.
	*
	* To avoid blocking, prefer `jsAsync()` when possible.
	*/
	js() {
		return dataToJs(this.dtype, this.dataSync(), this.shape);
	}
	/** Convert this array into a JavaScript object, asynchronously. */
	async jsAsync() {
		return dataToJs(this.dtype, await this.data(), this.shape);
	}
	/**
	* Copy an element of an array to a numeric scalar and return it.
	*
	* Throws an error if the array does not have a single element. The array must
	* either be rank-0, or all dimensions of the shape are 1.
	*/
	item() {
		if (this.size !== 1) throw new Error(`item() can only be called on arrays of size 1`);
		return this.dataSync()[0];
	}
	/** @private Internal plumbing method for Array / Tracer ops. */
	static _implRules() {
		return {
			[Primitive.Add]([x, y]) {
				return [x.#binary(require_backend.AluOp.Add, y)];
			},
			[Primitive.Mul]([x, y]) {
				return [x.#binary(require_backend.AluOp.Mul, y)];
			},
			[Primitive.Idiv]([x, y]) {
				return [x.#binary(require_backend.AluOp.Idiv, y)];
			},
			[Primitive.Mod]([x, y]) {
				return [x.#binary(require_backend.AluOp.Mod, y)];
			},
			[Primitive.Min]([x, y]) {
				return [x.#binary(require_backend.AluOp.Min, y)];
			},
			[Primitive.Max]([x, y]) {
				return [x.#binary(require_backend.AluOp.Max, y)];
			},
			[Primitive.Neg]([x]) {
				return [zerosLike$1(x).#binary(require_backend.AluOp.Sub, x)];
			},
			[Primitive.Reciprocal]([x]) {
				return [x.#unary(require_backend.AluOp.Reciprocal)];
			},
			[Primitive.Floor]([x]) {
				return [x.#unary(require_backend.AluOp.Floor)];
			},
			[Primitive.Ceil]([x]) {
				return [x.#unary(require_backend.AluOp.Ceil)];
			},
			[Primitive.StopGradient]([x]) {
				return [x];
			},
			[Primitive.Cast]([x], { dtype }) {
				return [x.#unary(require_backend.AluOp.Cast, dtype)];
			},
			[Primitive.Bitcast]([x], { dtype }) {
				if (x.dtype === require_backend.DType.Bool || dtype === require_backend.DType.Bool) throw new TypeError("Bitcast to/from bool is not allowed");
				if (x.dtype === dtype) return [x];
				if (require_backend.byteWidth(x.dtype) !== require_backend.byteWidth(dtype)) throw new TypeError(`Bitcast from ${x.dtype} to ${dtype} with different byte width`);
				if (x.#source instanceof require_backend.AluExp) return [x.#unary(require_backend.AluOp.Bitcast, dtype)];
				else {
					x.#backend.incRef(x.#source);
					const pending = x.#pending;
					for (const exe of pending) exe.updateRc(1);
					return [x.#newArrayFrom({
						dtype,
						weakType: false,
						pending
					})];
				}
			},
			[Primitive.Sin]([x]) {
				return [x.#unary(require_backend.AluOp.Sin)];
			},
			[Primitive.Cos]([x]) {
				return [x.#unary(require_backend.AluOp.Cos)];
			},
			[Primitive.Asin]([x]) {
				return [x.#unary(require_backend.AluOp.Asin)];
			},
			[Primitive.Atan]([x]) {
				return [x.#unary(require_backend.AluOp.Atan)];
			},
			[Primitive.Exp]([x]) {
				return [x.#unary(require_backend.AluOp.Exp)];
			},
			[Primitive.Log]([x]) {
				return [x.#unary(require_backend.AluOp.Log)];
			},
			[Primitive.Erf]([x]) {
				return [x.#unary(require_backend.AluOp.Erf)];
			},
			[Primitive.Erfc]([x]) {
				return [x.#unary(require_backend.AluOp.Erfc)];
			},
			[Primitive.Sqrt]([x]) {
				return [x.#unary(require_backend.AluOp.Sqrt)];
			},
			[Primitive.Reduce]([x], { op, axis }) {
				if (axis.length === 0) return [x];
				const moved = x.#moveAxesDown(axis);
				const result = moved.#reduce(op);
				moved.dispose();
				return [result];
			},
			[Primitive.Pool]([x], { window, strides }) {
				const st = pool(x.#st, window, strides);
				return [x.#reshape(st)];
			},
			[Primitive.PoolTranspose]([x], { inShape, window, strides }) {
				const n = inShape.length;
				let st = poolTranspose(x.#st, inShape, window, strides);
				st = st.reshape([...st.shape.slice(0, n), require_backend.prod(st.shape.slice(n))]);
				const reshaped = x.#reshape(st);
				const result = reshaped.#reduce(require_backend.AluOp.Add);
				reshaped.dispose();
				return [result];
			},
			[Primitive.Dot]([x, y]) {
				return [Array$1.#naryCustom("dot", ([x$1, y$1]) => require_backend.AluExp.mul(x$1, y$1), [x, y], { reduceAxis: true })];
			},
			[Primitive.Conv]([x, y], params) {
				checkConvShape(x.shape, y.shape, params);
				const [stX, stY] = prepareConv(x.#st, y.#st, params);
				const xView = x.#reshape(stX);
				const yView = y.#reshape(stY);
				const result = Array$1.#naryCustom("conv", ([x$1, y$1]) => require_backend.AluExp.mul(x$1, y$1), [xView, yView], { reduceAxis: true });
				xView.dispose();
				yView.dispose();
				return [result];
			},
			[Primitive.Compare]([x, y], { op }) {
				const custom = ([x$1, y$1]) => aluCompare(x$1, y$1, op);
				return [Array$1.#naryCustom("compare", custom, [x, y], { strongTypeOutput: true })];
			},
			[Primitive.Where]([cond, x, y]) {
				const custom = ([cond$1, x$1, y$1]) => require_backend.AluExp.where(cond$1, x$1, y$1);
				return [Array$1.#naryCustom("where", custom, [
					cond,
					x,
					y
				], { dtypeOverride: [require_backend.DType.Bool] })];
			},
			[Primitive.Concatenate](xs, { axis }) {
				const ndim$2 = xs[0].ndim;
				const sizes = xs.map((x) => x.shape[axis]);
				const finalSize = sizes.reduce((a, b) => a + b, 0);
				const makePadAxis = (start, end) => require_backend.range(ndim$2).map((i) => i === axis ? [start, end] : [0, 0]);
				let cum = 0;
				const xsPadded = [];
				for (let i = 0; i < xs.length; i++) {
					const padding = makePadAxis(cum, finalSize - cum - sizes[i]);
					xsPadded.push(xs[i].#reshape(xs[i].#st.pad(padding)));
					cum += sizes[i];
				}
				const custom = (exps) => exps.reduce(require_backend.AluExp.add);
				const result = Array$1.#naryCustom("concatenate", custom, xsPadded);
				for (const p of xsPadded) p.dispose();
				return [result];
			},
			[Primitive.Split]([x], { axis, sizes }) {
				const outputs = [];
				for (let i = 0, start = 0; i < sizes.length; i++) {
					const slice = require_backend.range(x.ndim).map((d) => d === axis ? [start, start + sizes[i]] : [0, x.shape[d]]);
					outputs.push(x.#reshape(x.#st.shrink(slice)));
					start += sizes[i];
				}
				return outputs;
			},
			[Primitive.RandomBits]([k0, k1], { shape: shape$1, mode }) {
				try {
					var _usingCtx3 = (0, import_usingCtx$10.default)();
					const keyShape = k0.shape;
					const genShape = shape$1.slice(keyShape.length);
					const c0 = _usingCtx3.u(zeros(genShape, {
						dtype: require_backend.DType.Uint32,
						device: k0.device
					}));
					const c1 = _usingCtx3.u(arange(0, require_backend.prod(genShape), 1, {
						dtype: require_backend.DType.Uint32,
						device: k0.device
					}).reshape(genShape));
					const k0View = k0.#reshape(k0.#st.reshape(keyShape.concat(require_backend.rep(genShape.length, 1))));
					const k1View = k1.#reshape(k1.#st.reshape(keyShape.concat(require_backend.rep(genShape.length, 1))));
					const custom = ([k0$1, k1$1, c0$1, c1$1]) => require_backend.AluExp.threefry2x32(k0$1, k1$1, c0$1, c1$1, mode);
					const result = Array$1.#naryCustom("random_bits", custom, [
						k0View,
						k1View,
						c0,
						c1
					]);
					k0View.dispose();
					k1View.dispose();
					return [result];
				} catch (_) {
					_usingCtx3.e = _;
				} finally {
					_usingCtx3.d();
				}
			},
			[Primitive.Gather]([x, ...indices], { axis, outDim }) {
				return [x.#gather(indices, axis, outDim)];
			},
			[Primitive.Transpose]([x], { perm }) {
				return [x.#transpose(perm)];
			},
			[Primitive.Broadcast]([x], { shape: shape$1, axis }) {
				return [x.#reshape(x.#st.broadcast(shape$1, axis))];
			},
			[Primitive.Reshape]([x], { shape: shape$1 }) {
				return [x.#reshape(x.#st.reshape(shape$1))];
			},
			[Primitive.Flip]([x], { axis }) {
				const arg = require_backend.rep(x.ndim, false);
				for (const ax of axis) arg[ax] = true;
				return [x.#reshape(x.#st.flip(arg))];
			},
			[Primitive.Shrink]([x], { slice }) {
				return [x.#reshape(x.#st.shrink(slice))];
			},
			[Primitive.Pad]([x], { width }) {
				return [x.#reshape(x.#st.pad(width))];
			},
			[Primitive.Sort]: Array$1.#routine(Primitive.Sort),
			[Primitive.Argsort]: Array$1.#routine(Primitive.Argsort),
			[Primitive.TriangularSolve]: Array$1.#routine(Primitive.TriangularSolve),
			[Primitive.Cholesky]: Array$1.#routine(Primitive.Cholesky),
			[Primitive.LU]: Array$1.#routine(Primitive.LU),
			[Primitive.Jit](args, { jaxpr, name }) {
				if (jaxpr.inBinders.length !== args.length) throw new Error(`jit expects ${jaxpr.inBinders.length} args, got ${args.length}`);
				const { backend, committed } = Array$1.#computeBackend("jit", args);
				args = args.map((ar) => ar._putSync(backend));
				const slots = args.map((x) => x._realizeSource());
				for (const ar of args) for (const exe of ar.#pending) {
					exe.prepareSync();
					exe.submit();
				}
				const jp = jitCompile(backend, jaxpr);
				const { outputs, pending } = jp.execute(slots);
				for (const exe of pending) exe.updateRc(+outputs.length - 1);
				return outputs.map((source, i) => {
					return new Array$1({
						source,
						st: require_backend.ShapeTracker.fromShape(jaxpr.outs[i].aval.shape),
						dtype: jaxpr.outs[i].aval.dtype,
						weakType: jaxpr.outs[i].aval.weakType,
						backend,
						committed,
						pending
					});
				});
			},
			[Primitive.DynamicUpdateSlice]([dst, src], { offset, axis }) {
				const dstShape = dst.shape;
				const srcShape = src.shape;
				if (dstShape.length === srcShape.length) {
					for (let i = 0; i < dstShape.length; i++) {
						if (i === axis) continue;
						if (dstShape[i] !== srcShape[i]) throw new Error("dynamicUpdateSlice: dst and src must match on non-updated axes");
					}
					if (offset + srcShape[axis] > dstShape[axis]) throw new Error("dynamicUpdateSlice: offset + src.shape[axis] out of bounds");
					const innerBefore = require_backend.prod(dstShape.slice(axis + 1));
					const outerBefore = require_backend.prod(dstShape.slice(0, axis));
					const dstData = dst.dataSync();
					const srcData = src.dataSync();
					for (let out = 0; out < outerBefore; out++) for (let i = 0; i < srcShape[axis]; i++) {
						const srcStart = (out * srcShape[axis] + i) * innerBefore;
						const dstStart = (out * dstShape[axis] + offset + i) * innerBefore;
						dstData.set(srcData.subarray(srcStart, srcStart + innerBefore), dstStart);
					}
					return [array(dstData, {
						shape: dstShape,
						dtype: dst.dtype,
						device: dst.device
					})];
				}
				if (axis === 0 && dstShape.length === srcShape.length + 1) {
					for (let i = 0; i < srcShape.length; i++) if (dstShape[i + 1] !== srcShape[i]) throw new Error("dynamicUpdateSlice: dst and src must match on non-updated axes (stacked mode)");
					if (offset + 1 > dstShape[0]) throw new Error("dynamicUpdateSlice: offset out of bounds for stacked dst");
					const { backend } = Array$1.#computeBackend("dynamicUpdateSlice", [dst, src]);
					if (backend.copyBufferToBuffer) {
						const dstSlot = dst._realizeSource();
						const srcSlot = src._realizeSource();
						const innerSizeBytes = require_backend.prod(srcShape) * require_backend.byteWidth(dst.dtype);
						const dstOffsetBytes = offset * innerSizeBytes;
						const canBufferCopy = dstOffsetBytes % 4 === 0 && innerSizeBytes % 4 === 0;
						if (canBufferCopy) {
							for (const p of dst.#pending) {
								p.prepareSync();
								p.submit();
							}
							for (const p of src.#pending) {
								p.prepareSync();
								p.submit();
							}
							backend.copyBufferToBuffer(srcSlot, 0, dstSlot, dstOffsetBytes, innerSizeBytes);
							backend.incRef(dstSlot);
							return [new Array$1({
								source: dstSlot,
								st: require_backend.ShapeTracker.fromShape(dstShape),
								dtype: dst.dtype,
								weakType: dst.weakType,
								backend,
								committed: true,
								pending: []
							})];
						}
					}
					const dstData = dst.dataSync();
					const srcData = src.dataSync();
					const innerSize = require_backend.prod(srcShape);
					const dstStart = offset * innerSize;
					dstData.set(srcData, dstStart);
					return [array(dstData, {
						shape: dstShape,
						dtype: dst.dtype,
						device: dst.device
					})];
				}
				throw new Error("dynamicUpdateSlice: unsupported dst/src shapes for update");
			},
			[Primitive.Scan](args, { jaxpr, numCarry, numConsts, length, reverse, acceptPath }) {
				args.slice(0, numConsts);
				args.slice(numConsts, numConsts + numCarry);
				const xs = args.slice(numConsts + numCarry);
				const numX = xs.length;
				const numY = jaxpr.outs.length - numCarry;
				if (jaxpr.inBinders.length !== numConsts + numCarry + numX) throw new Error(`scan jaxpr expects ${jaxpr.inBinders.length} inputs, got ${numConsts + numCarry + numX}`);
				const { backend, committed } = Array$1.#computeBackend("scan", args);
				const allArgs = args.map((ar) => ar._putSync(backend));
				const constArgs = allArgs.slice(0, numConsts);
				const initCarryArgs = allArgs.slice(numConsts, numConsts + numCarry);
				const xsArgs = allArgs.slice(numConsts + numCarry);
				const bodyProgram = jitCompile(backend, jaxpr);
				const plan = planScan(backend, bodyProgram, jaxpr, length, numCarry, numConsts, numX, numY, reverse, acceptPath);
				const constSlots = constArgs.map((c) => {
					const slot = c._realizeSource();
					for (const p of c.#pending) {
						p.prepareSync();
						p.submit();
					}
					return slot;
				});
				const initCarrySlots = initCarryArgs.map((c) => {
					const slot = c._realizeSource();
					for (const p of c.#pending) {
						p.prepareSync();
						p.submit();
					}
					return slot;
				});
				const xsSlots = xsArgs.map((x) => {
					const slot = x._realizeSource();
					for (const p of x.#pending) {
						p.prepareSync();
						p.submit();
					}
					return slot;
				});
				const xsAvals = xsArgs.map((x) => new ShapedArray(x.shape.slice(1), x.dtype, x.aval.weakType));
				const outputSlots = [];
				const bodyOutAvals = jaxpr.outs.map((v) => v.aval);
				for (let ci = 0; ci < numCarry; ci++) {
					const aval = bodyOutAvals[ci];
					const sizeBytes = require_backend.prod(aval.shape) * require_backend.byteWidth(aval.dtype);
					outputSlots.push(backend.malloc(sizeBytes > 0 ? sizeBytes : 1));
				}
				for (let yi = 0; yi < numY; yi++) {
					const aval = bodyOutAvals[numCarry + yi];
					const totalSizeBytes = length * require_backend.prod(aval.shape) * require_backend.byteWidth(aval.dtype);
					outputSlots.push(backend.malloc(totalSizeBytes > 0 ? totalSizeBytes : 1));
				}
				for (const s of constSlots) backend.incRef(s);
				for (const s of xsSlots) backend.incRef(s);
				const result = executeScan({
					backend,
					plan,
					bodyProgram,
					bodyJaxpr: jaxpr,
					length,
					numCarry,
					numConsts,
					numX,
					numY,
					reverse,
					constSlots,
					initCarrySlots,
					xsSlots,
					xsAvals,
					outputSlots
				});
				for (const p of result.pending) {
					p.prepareSync();
					p.submit();
				}
				for (const s of constSlots) backend.decRef(s);
				for (const s of xsSlots) backend.decRef(s);
				const outputs = [];
				for (let ci = 0; ci < numCarry; ci++) {
					const aval = bodyOutAvals[ci];
					outputs.push(new Array$1({
						source: result.outputs[ci],
						st: require_backend.ShapeTracker.fromShape(aval.shape),
						dtype: aval.dtype,
						weakType: aval.weakType,
						backend,
						committed,
						pending: []
					}));
				}
				for (let yi = 0; yi < numY; yi++) {
					const aval = bodyOutAvals[numCarry + yi];
					outputs.push(new Array$1({
						source: result.outputs[numCarry + yi],
						st: require_backend.ShapeTracker.fromShape([length, ...aval.shape]),
						dtype: aval.dtype,
						weakType: aval.weakType,
						backend,
						committed,
						pending: []
					}));
				}
				return outputs;
			}
		};
	}
	/** @private */
	_realizeSource() {
		this.#realize();
		return this.#source;
	}
	/**
	* Submit all pending backend dispatches without reading data back.
	* Used by evalJaxprTransposed to flush result arrays' PE chains before
	* disposing intermediates — prevents orphaned Slot references.
	* @private
	*/
	_flushPendingSync() {
		this.#realize();
		for (const p of this.#pending) {
			p.prepareSync();
			p.submit();
		}
	}
	/** @private Put this array on a new backend, asynchronously. */
	async _put(backend) {
		if (this.#backend === backend) return this;
		if (this.#source instanceof require_backend.AluExp) return this.#newArrayFrom({
			backend,
			committed: true
		});
		else {
			const data = await this.data();
			return arrayFromData(data, this.shape, {
				dtype: this.#dtype,
				device: backend.type
			}, this.#weakType);
		}
	}
	/** @private Put this array on a new backend, synchronously. */
	_putSync(backend) {
		if (this.#backend === backend) return this;
		if (this.#source instanceof require_backend.AluExp) return this.#newArrayFrom({
			backend,
			committed: true
		});
		else {
			const data = this.dataSync();
			return arrayFromData(data, this.shape, {
				dtype: this.#dtype,
				device: backend.type
			}, this.#weakType);
		}
	}
};
/** Constructor for creating a new array from data. */
function array(values, { shape: shape$1, dtype, device } = {}) {
	if (values instanceof Tracer) {
		if (shape$1 && !require_backend.deepEqual(values.shape, shape$1)) values = values.reshape(shape$1);
		if (dtype && values.dtype !== dtype) values = values.astype(dtype);
		return values;
	} else if (ArrayBuffer.isView(values)) return markAnonymousIfTracing(arrayFromData(values, shape$1 ?? [values.length], {
		dtype,
		device
	}));
	else {
		if (!shape$1) {
			shape$1 = [];
			let cur = values;
			while (JsArray$1.isArray(cur)) {
				shape$1.push(cur.length);
				cur = cur[0];
			}
		}
		const size$1 = require_backend.prod(shape$1);
		const flat = require_backend.recursiveFlatten(values);
		if (flat.length !== size$1) throw new Error(`Jagged shape: ${JSON.stringify(shape$1)} vs ${flat.length}`);
		if (size$1 === 0) return markAnonymousIfTracing(zeros(shape$1, {
			dtype,
			device
		}));
		if (size$1 === 1) return markAnonymousIfTracing(full(shape$1, flat[0], {
			dtype,
			device
		}));
		if (typeof flat[0] === "boolean") {
			dtype = dtype ?? require_backend.DType.Bool;
			const data = new Int32Array(flat.map((x) => x ? 1 : 0));
			return markAnonymousIfTracing(arrayFromData(data, shape$1, {
				dtype,
				device
			}));
		} else {
			const weakType = dtype == void 0 && shape$1.length === 0;
			dtype = dtype ?? require_backend.DType.Float32;
			const data = require_backend.dtypedJsArray(dtype, flat);
			return markAnonymousIfTracing(arrayFromData(data, shape$1, {
				dtype,
				device
			}, weakType));
		}
	}
}
function arrayFromData(data, shape$1, { dtype, device }, weakType = false) {
	if (data instanceof Float32Array) {
		if (dtype && dtype !== require_backend.DType.Float32) throw new Error("Float32Array must have float32 type");
		dtype ??= require_backend.DType.Float32;
	} else if (data instanceof Int32Array) {
		if (dtype && dtype !== require_backend.DType.Int32 && dtype !== require_backend.DType.Bool) throw new Error("Int32Array must have int32 or bool type");
		dtype ??= require_backend.DType.Int32;
	} else if (data instanceof Uint32Array) {
		if (dtype && dtype !== require_backend.DType.Uint32) throw new Error("Uint32Array must have uint32 type");
		dtype ??= require_backend.DType.Uint32;
	} else if (data instanceof Float16Array) {
		if (dtype && dtype !== require_backend.DType.Float16) throw new Error("Float16Array must have float16 type");
		dtype ??= require_backend.DType.Float16;
	} else if (data instanceof Float64Array) {
		if (dtype && dtype !== require_backend.DType.Float64) throw new Error("Float64Array must have float64 type");
		dtype ??= require_backend.DType.Float64;
	} else throw new Error("Unsupported data array type: " + data.constructor.name);
	if (data.length < inlineArrayLimit) {
		let allEqual = true;
		for (let i = 1; i < data.length; i++) if (data[i] !== data[0]) {
			allEqual = false;
			break;
		}
		if (allEqual) {
			const sa = new ShapedArray(shape$1, dtype, weakType);
			return fullInternal(sa, data[0], device);
		}
	}
	const backend = require_backend.getBackend(device);
	const buf = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
	const slot = backend.malloc(data.byteLength, buf);
	return new Array$1({
		source: slot,
		st: require_backend.ShapeTracker.fromShape(shape$1),
		dtype,
		weakType,
		backend,
		committed: device != void 0
	});
}
function dataToJs(dtype, data, shape$1) {
	if (shape$1.length === 0) return dtype === require_backend.DType.Bool ? Boolean(data[0]) : data[0];
	const [first, ...rest] = shape$1;
	const restSize = require_backend.prod(rest);
	const ret = [];
	for (let i = 0; i < first; i++) {
		const subarray = data.slice(i * restSize, (i + 1) * restSize);
		ret.push(dataToJs(dtype, subarray, rest));
	}
	return ret;
}
/** If x is a value, lift it into an array, otherwise leave it be. */
/**
* WeakSet tracking arrays created by pureArray from raw (non-Tracer) values.
* These arrays are "anonymous" — created transiently by fudgeArray/pureArray
* inside library functions, with no user code holding a reference.
* getOrMakeConstTracer checks this to avoid calling .ref on anonymous consts,
* which would leak when the ClosedJaxpr is disposed.
*/
const anonymousConstArrays = /* @__PURE__ */ new WeakSet();
/**
* Tags an array as anonymous builder-owned if we're inside a scan body trace.
* This prevents getOrMakeConstTracer from calling .ref on inline np.array()
* constants, which would leak when the ClosedJaxpr is disposed.
*/
const markAnonymousIfTracing = (arr) => {
	if (isScanBodyTraceActive()) anonymousConstArrays.add(arr);
	return arr;
};
function pureArray(x) {
	if (x instanceof Tracer) return x;
	else {
		const arr = array(x);
		anonymousConstArrays.add(arr);
		return arr;
	}
}
var EvalTrace = class extends Trace {
	pure = (x) => pureArray(x);
	lift = (x) => x;
	processPrimitive(primitive, tracers, params) {
		return implRules[primitive](tracers, params);
	}
};
const baseArrayTrace = new EvalTrace(newMain(EvalTrace, null));
const implRules = Array$1._implRules();
function fullInternal(aval, fillValue, device) {
	return new Array$1({
		source: require_backend.AluExp.const(aval.dtype, fillValue),
		st: require_backend.ShapeTracker.fromShape(aval.shape),
		dtype: aval.dtype,
		weakType: aval.weakType,
		backend: require_backend.getBackend(device),
		committed: device != void 0
	});
}
function zerosLike$1(val, opts) {
	return fullLike(val, 0, opts);
}
function onesLike$1(val, opts) {
	return fullLike(val, 1, opts);
}
function fullLike(val, fillValue, { dtype, shape: shape$1, device } = {}) {
	const aval = getAval(val);
	if (fillValue instanceof Tracer) throw new Error("numpy.fullLike() with array argument not implemented yet");
	const sa = new ShapedArray(shape$1 ?? aval.shape, dtype ?? aval.dtype, aval.weakType && dtype === void 0);
	return fullInternal(sa, fillValue, device);
}
/** Return a new array of given shape and type, filled with zeros. */
function zeros(shape$1, opts) {
	return full(shape$1, 0, opts);
}
/** Return a new array of given shape and type, filled with ones. */
function ones(shape$1, opts) {
	return full(shape$1, 1, opts);
}
/** Return a new array of given shape and type, filled with `fill_value`. */
function full(shape$1, fillValue, { dtype, device } = {}) {
	let weakType = dtype == void 0 && shape$1.length === 0;
	if (typeof fillValue === "number") dtype = dtype ?? require_backend.DType.Float32;
	else if (typeof fillValue === "boolean") {
		dtype = dtype ?? require_backend.DType.Bool;
		weakType = false;
	} else if (fillValue instanceof Tracer) throw new Error("numpy.full() with array argument not implemented yet");
	else throw new TypeError(`Invalid type for full: ${fillValue}`);
	return fullInternal(new ShapedArray(shape$1, dtype, weakType), fillValue, device);
}
/**
* Create an identity matrix.
*
* If numCols is not provided, it defaults to numRows, i.e., a square identity
* matrix with ones on the diagonal.
*/
function eye(numRows, numCols, { dtype, device } = {}) {
	numCols = numCols ?? numRows;
	const weakType = dtype == void 0;
	dtype = dtype ?? require_backend.DType.Float32;
	if (numCols < numRows) {
		const arr = eye(numCols, numRows, {
			dtype,
			device
		});
		return arr.transpose();
	}
	if (numRows === 0) return zeros([0, numCols], {
		dtype,
		device
	});
	const exp$2 = require_backend.AluExp.cmplt(require_backend.AluExp.mod(require_backend.AluVar.idx, require_backend.AluExp.i32(numCols + 1)), require_backend.AluExp.i32(1));
	return new Array$1({
		source: require_backend.AluExp.cast(dtype, exp$2),
		st: require_backend.ShapeTracker.fromShape([numRows, numCols]),
		dtype,
		weakType,
		backend: require_backend.getBackend(device),
		committed: device != void 0
	});
}
/** Return the identity matrix, with ones on the main diagonal. */
function identity$1(n, { dtype, device } = {}) {
	return eye(n, n, {
		dtype,
		device
	});
}
/**
* Return evenly spaced values within a given interval.
*
* This can be called with a varying number of arguments, just like the range()
* builtin function in Python.
*
* - `arange(stop)` is equivalent to `arange(0, stop, 1)`.
* - `arange(start, stop)` is equivalent to `arange(start, stop, 1)`.
* - `arange(start, stop, step)` creates an array starting at `start`, ending
*   before `stop`, with a step size of `step`.
*
* Defaults to an integer data type. This can produce unintended results when
* using a non-integer step, so prefer linspace() in those cases.
*/
function arange(start, stop, step = 1, { dtype, device } = {}) {
	dtype = dtype ?? require_backend.DType.Int32;
	if (stop === void 0) {
		stop = start;
		start = 0;
	}
	if (step === 0) throw new RangeError(`Invalid step for arange: ${step}. Step must be non-zero.`);
	const size$1 = Math.max(0, Math.ceil((stop - start) / step));
	if (size$1 === 0) return zeros([0], {
		dtype,
		device
	});
	const exp$2 = require_backend.AluExp.add(require_backend.AluExp.const(dtype, start), require_backend.AluExp.mul(require_backend.AluExp.cast(dtype, require_backend.AluVar.idx), require_backend.AluExp.const(dtype, step)));
	const st = require_backend.ShapeTracker.fromShape([size$1]);
	return new Array$1({
		source: exp$2,
		st,
		dtype,
		weakType: false,
		backend: require_backend.getBackend(device),
		committed: device != void 0
	});
}
/**
* Return an array with ones on and below the diagonal and zeros elsewhere.
*
* If `k` is provided, it specifies the sub-diagonal on and below which the
* array is filled with ones. `k=0` is the main diagonal, `k<0` is below it, and
* `k>0` is above it.
*/
function tri(n, m, k = 0, { dtype, device } = {}) {
	try {
		var _usingCtx4 = (0, import_usingCtx$10.default)();
		m ??= n;
		dtype ??= require_backend.DType.Float32;
		if (!Number.isInteger(n) || n < 0) throw new Error(`tri: n must be a non-negative integer, got ${n}`);
		if (!Number.isInteger(m) || m < 0) throw new Error(`tri: m must be a non-negative integer, got ${m}`);
		if (!Number.isInteger(k)) throw new Error(`tri: k must be an integer, got ${k}`);
		const rows = _usingCtx4.u(arange(k, n + k, 1, {
			dtype: require_backend.DType.Int32,
			device
		}));
		const cols = _usingCtx4.u(arange(0, m, 1, {
			dtype: require_backend.DType.Int32,
			device
		}));
		const rowsReshaped = _usingCtx4.u(rows.reshape([n, 1]));
		const ge = rowsReshaped.greaterEqual(cols);
		if (ge.dtype === dtype) return ge;
		_usingCtx4.u(ge);
		return ge.astype(dtype);
	} catch (_) {
		_usingCtx4.e = _;
	} finally {
		_usingCtx4.d();
	}
}
/** Return the lower triangle of an array. Must be of dimension >= 2. */
function tril(a, k = 0) {
	try {
		var _usingCtx5 = (0, import_usingCtx$10.default)();
		if (ndim$1(a) < 2) throw new Error(`tril: input array must be at least 2D, got ${ndim$1(a)}D`);
		a = fudgeArray(a);
		const [n, m] = a.shape.slice(-2);
		const mask = _usingCtx5.u(tri(n, m, k, { dtype: require_backend.DType.Bool }));
		const zeros$1 = _usingCtx5.u(zerosLike$1(a));
		return where$1(mask, a, zeros$1);
	} catch (_) {
		_usingCtx5.e = _;
	} finally {
		_usingCtx5.d();
	}
}
/** Return the upper triangle of an array. Must be of dimension >= 2. */
function triu(a, k = 0) {
	try {
		var _usingCtx6 = (0, import_usingCtx$10.default)();
		if (ndim$1(a) < 2) throw new Error(`tril: input array must be at least 2D, got ${ndim$1(a)}D`);
		a = fudgeArray(a);
		const [n, m] = a.shape.slice(-2);
		const mask = _usingCtx6.u(tri(n, m, k - 1, { dtype: require_backend.DType.Bool }));
		const zeros$1 = _usingCtx6.u(zerosLike$1(a));
		return where$1(mask, zeros$1, a);
	} catch (_) {
		_usingCtx6.e = _;
	} finally {
		_usingCtx6.d();
	}
}
/**
* Return evenly spaced numbers over a specified interval.
*
* Returns _num_ evenly spaced samples, calculated over the interval
* [`start`, `stop`]. The endpoint `stop` is included in the result by default,
* but this is controlled by the `endpoint` parameter.
*
* The default data type is Float32. Use arange() for integer steps.
*/
function linspace(start, stop, num = 50, endpoint = true, { dtype, device } = {}) {
	dtype = dtype ?? require_backend.DType.Float32;
	if (num < 0 || !Number.isInteger(num)) throw new RangeError(`Invalid num for linspace: ${num}. Must be non-negative integer.`);
	else if (num === 0) return zeros([0], {
		dtype,
		device
	});
	else if (num === 1) return full([1], start, {
		dtype,
		device
	});
	else if (start === stop) return full([num], start, {
		dtype,
		device
	});
	const delta = stop - start;
	const denom = endpoint ? num - 1 : num;
	const exp$2 = require_backend.AluExp.cast(dtype, require_backend.AluExp.add(require_backend.AluExp.f32(start), require_backend.AluExp.mul(require_backend.AluExp.f32(delta / denom), require_backend.AluExp.cast(require_backend.DType.Float32, require_backend.AluVar.idx))));
	const st = require_backend.ShapeTracker.fromShape([num]);
	return new Array$1({
		source: exp$2,
		st,
		dtype,
		weakType: false,
		backend: require_backend.getBackend(device),
		committed: device != void 0
	});
}
/**
* Return numbers spaced evenly on a log scale.
*
* In linear space, the sequence starts at `base ** start` and ends at
* `base ** stop` (see `endpoint` below).
*
* @param start - `base ** start` is the starting value of the sequence.
* @param stop - `base ** stop` is the final value of the sequence, unless `endpoint` is false.
* @param num - Number of samples to generate. Default is 50.
* @param endpoint - If true, `stop` is the last sample. Otherwise, it is not included. Default is true.
* @param base - The base of the log space. Default is 10.
* @returns Array of evenly spaced values on a log scale.
*/
function logspace(start, stop, num = 50, endpoint = true, base = 10, { dtype, device } = {}) {
	const y = linspace(start, stop, num, endpoint, {
		dtype,
		device
	});
	const logBase = Math.log(base);
	return exp$1(mul(y, logBase));
}
function aluCompare(a, b, op) {
	switch (op) {
		case CompareOp.Less: return require_backend.AluExp.cmplt(a, b);
		case CompareOp.Equal: return require_backend.AluExp.cmpne(a, b).not();
		case CompareOp.NotEqual: return require_backend.AluExp.cmpne(a, b);
		case CompareOp.LessEqual: return require_backend.AluExp.add(require_backend.AluExp.cmplt(a, b), require_backend.AluExp.cmpne(a, b).not());
	}
}

//#endregion
//#region src/frontend/vmap.ts
var import_usingCtx$9 = /* @__PURE__ */ __toESM(require_usingCtx(), 1);
function mappedAval(batchDim, aval) {
	const shape$1 = [...aval.shape];
	shape$1.splice(batchDim, 1);
	return new ShapedArray(shape$1, aval.dtype, aval.weakType);
}
/** Move one axis to a different index. */
function moveaxis(x, src, dst) {
	const t = pureArray(x);
	src = require_backend.checkAxis(src, t.ndim);
	dst = require_backend.checkAxis(dst, t.ndim);
	if (src === dst) return t;
	const perm = require_backend.range(t.ndim);
	perm.splice(src, 1);
	perm.splice(dst, 0, src);
	return transpose$1(t, perm);
}
function moveBatchAxis(axisSize, src, dst, x) {
	if (src === null) {
		const targetShape = [...x.shape];
		targetShape.splice(dst, 0, axisSize);
		return broadcast(x, targetShape, [dst]);
	} else if (src === dst) return x;
	else return moveaxis(x, src, dst);
}
var BatchTracer = class extends Tracer {
	constructor(trace$1, val, batchDim) {
		super(trace$1);
		this.val = val;
		this.batchDim = batchDim;
	}
	get aval() {
		if (this.batchDim === null) return this.val.aval;
		else return mappedAval(this.batchDim, this.val.aval);
	}
	toString() {
		return `BatchTracer(${this.val.toString()}, ${this.batchDim})`;
	}
	get ref() {
		this.val.ref;
		return this;
	}
	dispose() {
		this.val.dispose();
	}
	fullLower() {
		if (this.batchDim === null) return this.val.fullLower();
		else return this;
	}
};
var BatchTrace = class extends Trace {
	pure(val) {
		const arr = pureArray(val);
		if (!(val instanceof Tracer)) this.pureIntermediates.push(arr);
		return this.lift(arr);
	}
	lift(val) {
		return new BatchTracer(this, val, null);
	}
	processPrimitive(primitive, tracers, params) {
		const [valsIn, bdimsIn] = require_backend.unzip2(tracers.map((t) => [t.val, t.batchDim]));
		const vmapRule = vmapRules[primitive];
		if (vmapRule === void 0) throw new Error(`No vmap rule for: ${primitive}`);
		if (bdimsIn.every((d) => d === null)) {
			const valOuts$1 = bind(primitive, valsIn, params);
			const outs$1 = valOuts$1.map((x) => new BatchTracer(this, x, null));
			this.intermediates.push(...outs$1);
			return outs$1;
		}
		const [valOuts, bdimOuts] = vmapRule(this.axisSize, valsIn, bdimsIn, params);
		if (valOuts.length !== bdimOuts.length) throw new Error(`vmap rule for ${primitive} returned mismatched lengths: ${valOuts.length} vs ${bdimOuts.length}`);
		const outs = require_backend.zip(valOuts, bdimOuts).map(([x, bd]) => new BatchTracer(this, x, bd));
		this.intermediates.push(...outs);
		return outs;
	}
	get axisSize() {
		return this.main.globalData.axisSize;
	}
	get pureIntermediates() {
		return this.main.globalData.pureIntermediates;
	}
	get intermediates() {
		return this.main.globalData.intermediates;
	}
};
/**
* Process a primitive with built-in broadcasting.
*
* Reference: https://github.com/jax-ml/jax/blob/jax-v0.8.1/jax/_src/interpreters/batching.py#L1029
*/
function broadcastBatcher(prim) {
	return (axisSize, args, dims, params) => {
		if (args.length === 0) throw new Error("Empty list in broadcastBatcher");
		const nd = Math.max(...args.map((x, i) => ndim$1(x) + (dims[i] === null ? 1 : 0)));
		const firstIdx = dims.findIndex((d) => d !== null);
		const firstBdim = dims[firstIdx] - args[firstIdx].ndim;
		if (require_backend.zip(args, dims).every(([x, d]) => d === null && ndim$1(x) < -firstBdim || d !== null && d - x.ndim === firstBdim)) return [[bind1(prim, args, params)], [nd + firstBdim]];
		const origArgs = args;
		args = args.map((x, i) => {
			if (dims[i] === null) return x;
			const moved = moveBatchAxis(axisSize, dims[i], 0, x);
			if (moved.ndim < nd) {
				const reshaped = moved.reshape([
					moved.shape[0],
					...require_backend.rep(nd - moved.ndim, 1),
					...moved.shape.slice(1)
				]);
				if (moved !== x) moved[Symbol.dispose]();
				return reshaped;
			}
			return moved;
		});
		const result = bind1(prim, args, params);
		for (let i = 0; i < args.length; i++) if (args[i] !== origArgs[i]) args[i][Symbol.dispose]();
		return [[result], [0]];
	};
}
function unopBatcher(prim) {
	return (axisSize, [x], [xBdim], params) => {
		return [[bind1(prim, [x], params)], [xBdim]];
	};
}
function lastDimsBatcher(prim, inputDims, numOutputs = 1) {
	return (axisSize, [x], [xBdim], params) => {
		require_backend.assertNonNull(xBdim);
		if (xBdim < x.ndim - inputDims) return [bind(prim, [x], params), require_backend.rep(numOutputs, xBdim)];
		const origX = x;
		x = moveBatchAxis(axisSize, xBdim, 0, x);
		const result = bind(prim, [x], params);
		if (x !== origX) x[Symbol.dispose]();
		return [result, require_backend.rep(numOutputs, 0)];
	};
}
const vmapRules = {
	[Primitive.Add]: broadcastBatcher(Primitive.Add),
	[Primitive.Mul]: broadcastBatcher(Primitive.Mul),
	[Primitive.Idiv]: broadcastBatcher(Primitive.Idiv),
	[Primitive.Mod]: broadcastBatcher(Primitive.Mod),
	[Primitive.Min]: broadcastBatcher(Primitive.Min),
	[Primitive.Max]: broadcastBatcher(Primitive.Max),
	[Primitive.Neg]: unopBatcher(Primitive.Neg),
	[Primitive.Reciprocal]: unopBatcher(Primitive.Reciprocal),
	[Primitive.Floor]: unopBatcher(Primitive.Floor),
	[Primitive.Ceil]: unopBatcher(Primitive.Ceil),
	[Primitive.StopGradient]: unopBatcher(Primitive.StopGradient),
	[Primitive.Cast]: unopBatcher(Primitive.Cast),
	[Primitive.Bitcast]: unopBatcher(Primitive.Bitcast),
	[Primitive.Sin]: unopBatcher(Primitive.Sin),
	[Primitive.Cos]: unopBatcher(Primitive.Cos),
	[Primitive.Asin]: unopBatcher(Primitive.Asin),
	[Primitive.Atan]: unopBatcher(Primitive.Atan),
	[Primitive.Exp]: unopBatcher(Primitive.Exp),
	[Primitive.Log]: unopBatcher(Primitive.Log),
	[Primitive.Erf]: unopBatcher(Primitive.Erf),
	[Primitive.Erfc]: unopBatcher(Primitive.Erfc),
	[Primitive.Sqrt]: unopBatcher(Primitive.Sqrt),
	[Primitive.Reduce](axisSize, [x], [xBdim], { op, axis }) {
		require_backend.assertNonNull(xBdim);
		const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
		const outBdim = xBdim - axis.filter((ax) => ax < xBdim).length;
		return [[reduce(x, op, newAxis)], [outBdim]];
	},
	[Primitive.Dot](axisSize, [x, y], [xBdim, yBdim]) {
		const origX = x, origY = y;
		x = moveBatchAxis(axisSize, xBdim, x.ndim - (xBdim === null ? 1 : 2), x);
		y = moveBatchAxis(axisSize, yBdim, y.ndim - (yBdim === null ? 1 : 2), y);
		const z = dot$2(x, y);
		if (x !== origX) x[Symbol.dispose]();
		if (y !== origY) y[Symbol.dispose]();
		return [[z], [z.ndim - 1]];
	},
	[Primitive.Conv](axisSize, [x, y], [xBdim, yBdim], params) {
		const origX = x, origY = y;
		x = moveBatchAxis(axisSize, xBdim, 0, x);
		y = moveBatchAxis(axisSize, yBdim, 0, y);
		const z = conv$1(x, y, {
			...params,
			vmapDims: params.vmapDims + 1
		});
		if (x !== origX) x[Symbol.dispose]();
		if (y !== origY) y[Symbol.dispose]();
		return [[z], [0]];
	},
	[Primitive.Compare]: broadcastBatcher(Primitive.Compare),
	[Primitive.Where]: broadcastBatcher(Primitive.Where),
	[Primitive.Concatenate](axisSize, xs, xBdims, { axis }) {
		const minBdim = Math.min(...xBdims.filter((d) => d !== null));
		const origXs = xs;
		xs = xs.map((x, i) => moveBatchAxis(axisSize, xBdims[i], minBdim, x));
		const newAxis = axis + (minBdim <= axis ? 1 : 0);
		const result = concatenate$1(xs, newAxis);
		for (let i = 0; i < xs.length; i++) if (xs[i] !== origXs[i]) xs[i][Symbol.dispose]();
		return [[result], [minBdim]];
	},
	[Primitive.Split](axisSize, [x], [xBdim], { axis, sizes }) {
		require_backend.assertNonNull(xBdim);
		const newAxis = axis + (xBdim <= axis ? 1 : 0);
		const outs = split$2(x, newAxis, sizes);
		return [outs, require_backend.rep(outs.length, xBdim)];
	},
	[Primitive.RandomBits](axisSize, [k0, k1], [bdim0, bdim1], { shape: shape$1, mode }) {
		const origK0 = k0, origK1 = k1;
		k0 = moveBatchAxis(axisSize, bdim0, 0, k0);
		k1 = moveBatchAxis(axisSize, bdim1, 0, k1);
		const result = randomBits(k0, k1, [axisSize, ...shape$1], mode);
		if (k0 !== origK0) k0[Symbol.dispose]();
		if (k1 !== origK1) k1[Symbol.dispose]();
		return [[result], [0]];
	},
	[Primitive.Gather](axisSize, [x, ...indices], [xBdim, ...indicesBdim], { axis, outDim }) {
		if (indicesBdim.every((d) => d === null)) {
			require_backend.assertNonNull(xBdim);
			const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
			let newBdim = xBdim - axis.filter((ax) => ax < xBdim).length;
			let newOutDim = outDim;
			if (newOutDim < newBdim) newBdim += axis.length;
			else newOutDim += 1;
			return [[gather(x, indices, newAxis, newOutDim)], [newBdim]];
		}
		const nd = Math.max(...indices.map((m, i) => ndim$1(m) + (indicesBdim[i] === null ? 1 : 0)));
		const origIndices = [...indices];
		indices = indices.map((m, i) => {
			if (indicesBdim[i] === null) return m;
			const moved = moveBatchAxis(axisSize, indicesBdim[i], 0, m);
			if (moved.ndim < nd) {
				const reshaped = moved.reshape([
					moved.shape[0],
					...require_backend.rep(nd - moved.ndim, 1),
					...moved.shape.slice(1)
				]);
				if (moved !== m) moved[Symbol.dispose]();
				return reshaped;
			}
			return moved;
		});
		if (xBdim === null) {
			const result = gather(x, indices, axis, outDim);
			for (let i = 0; i < indices.length; i++) if (indices[i] !== origIndices[i]) indices[i][Symbol.dispose]();
			return [[result], [outDim]];
		} else {
			const origX = x;
			x = moveBatchAxis(axisSize, xBdim, 0, x);
			const newAxis = [0, ...axis.map((ax) => ax + 1)];
			const arangeArr = arange(axisSize);
			const extraBatchIndex = arangeArr.reshape([-1, ...require_backend.rep(nd - 1, 1)]);
			arangeArr.dispose();
			indices.splice(0, 0, extraBatchIndex);
			const result = gather(x, indices, newAxis, outDim);
			if (x !== origX) x.dispose();
			extraBatchIndex.dispose();
			for (let i = 1; i < indices.length; i++) if (indices[i] !== origIndices[i - 1]) indices[i].dispose();
			return [[result], [outDim]];
		}
	},
	[Primitive.Transpose](axisSize, [x], [xBdim], { perm }) {
		require_backend.assertNonNull(xBdim);
		const newPerm = perm.map((p) => p + (xBdim <= p ? 1 : 0));
		newPerm.splice(xBdim, 0, xBdim);
		return [[transpose$1(x, newPerm)], [xBdim]];
	},
	[Primitive.Broadcast](axisSize, [x], [xBdim], { shape: shape$1, axis }) {
		require_backend.assertNonNull(xBdim);
		const newShape = shape$1.toSpliced(xBdim, 0, axisSize);
		const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
		return [[broadcast(x, newShape, newAxis)], [xBdim]];
	},
	[Primitive.Reshape](axisSize, [x], [xBdim], { shape: shape$1 }) {
		const origX = x;
		x = moveBatchAxis(axisSize, xBdim, 0, x);
		const result = reshape$1(x, [axisSize, ...shape$1]);
		if (x !== origX) x[Symbol.dispose]();
		return [[result], [0]];
	},
	[Primitive.Flip](axisSize, [x], [xBdim], { axis }) {
		require_backend.assertNonNull(xBdim);
		const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
		return [[flip$1(x, newAxis)], [xBdim]];
	},
	[Primitive.Shrink](axisSize, [x], [xBdim], { slice }) {
		require_backend.assertNonNull(xBdim);
		const newSlice = slice.toSpliced(xBdim, 0, [0, axisSize]);
		return [[shrink(x, newSlice)], [xBdim]];
	},
	[Primitive.Pad](axisSize, [x], [xBdim], { width }) {
		require_backend.assertNonNull(xBdim);
		const newWidth = width.toSpliced(xBdim, 0, [0, 0]);
		return [[pad$1(x, newWidth)], [xBdim]];
	},
	[Primitive.Sort]: lastDimsBatcher(Primitive.Sort, 1),
	[Primitive.Argsort]: lastDimsBatcher(Primitive.Argsort, 1, 2),
	[Primitive.TriangularSolve](axisSize, [a, b], [aBdim, bBdim], { unitDiagonal }) {
		if (aBdim === null) {
			const origB$1 = b;
			b = moveBatchAxis(axisSize, bBdim, -3, b);
			const bMoved = b;
			const [s, m, n] = b.shape.slice(-3);
			b = b.reshape([
				...b.shape.slice(0, -3),
				s * m,
				n
			]);
			if (bMoved !== origB$1) bMoved[Symbol.dispose]();
			let x$1 = bind1(Primitive.TriangularSolve, [a, b], { unitDiagonal });
			const xOrig = x$1;
			const savedShape = [...b.shape.slice(0, -2)];
			x$1 = x$1.reshape([
				...savedShape,
				s,
				m,
				n
			]);
			if (b !== origB$1) b[Symbol.dispose]();
			xOrig[Symbol.dispose]();
			return [[x$1], [x$1.ndim - 3]];
		}
		const origA = a, origB = b;
		a = moveBatchAxis(axisSize, aBdim, 0, a);
		b = moveBatchAxis(axisSize, bBdim, 0, b);
		const x = bind1(Primitive.TriangularSolve, [a, b], { unitDiagonal });
		if (a !== origA) a[Symbol.dispose]();
		if (b !== origB) b[Symbol.dispose]();
		return [[x], [0]];
	},
	[Primitive.Cholesky]: lastDimsBatcher(Primitive.Cholesky, 2),
	[Primitive.LU]: lastDimsBatcher(Primitive.LU, 2, 3),
	[Primitive.Jit](axisSize, args, dims, { name, jaxpr }) {
		const newJaxpr = vmapJaxpr(jaxpr, axisSize, dims);
		const outs = bind(Primitive.Jit, [...newJaxpr.consts, ...args], {
			name: `${name}_vmap`,
			jaxpr: newJaxpr.jaxpr,
			numConsts: newJaxpr.consts.length
		});
		return [outs, require_backend.rep(outs.length, 0)];
	},
	[Primitive.DynamicUpdateSlice]() {
		throw new Error("DynamicUpdateSlice vmap: not yet implemented");
	},
	[Primitive.Scan](axisSize, args, dims, { jaxpr, numCarry, numConsts, length, reverse }) {
		const numX = args.length - numConsts - numCarry;
		const numY = jaxpr.outs.length - numCarry;
		const consts = args.slice(0, numConsts);
		const initCarry = args.slice(numConsts, numConsts + numCarry);
		const xs = args.slice(numConsts + numCarry);
		const constDims = dims.slice(0, numConsts);
		const carryDims = dims.slice(numConsts, numConsts + numCarry);
		const xsDims = dims.slice(numConsts + numCarry);
		const movedConsts = consts.map((c, i) => moveBatchAxis(axisSize, constDims[i], 0, c));
		const movedCarry = initCarry.map((c, i) => moveBatchAxis(axisSize, carryDims[i], 0, c));
		const movedXs = xs.map((x, i) => {
			if (xsDims[i] === null) {
				const newShape = [
					x.shape[0],
					axisSize,
					...x.shape.slice(1)
				];
				return broadcast(x, newShape, [1]);
			} else if (xsDims[i] === 0) return moveaxis(x, 0, 1);
			else return moveBatchAxis(axisSize, xsDims[i], 1, x);
		});
		const bodyDims = [
			...require_backend.rep(numConsts, 0),
			...require_backend.rep(numCarry, 0),
			...require_backend.rep(numX, 0)
		];
		const vmappedBody = vmapJaxpr(jaxpr, axisSize, bodyDims);
		const scanArgs = [
			...vmappedBody.consts,
			...movedConsts,
			...movedCarry,
			...movedXs
		];
		const results = bind(Primitive.Scan, scanArgs, {
			jaxpr: vmappedBody.jaxpr,
			numCarry,
			numConsts: vmappedBody.consts.length,
			length,
			reverse
		});
		for (let i = 0; i < movedConsts.length; i++) if (movedConsts[i] !== consts[i]) movedConsts[i][Symbol.dispose]();
		for (let i = 0; i < movedCarry.length; i++) if (movedCarry[i] !== initCarry[i]) movedCarry[i][Symbol.dispose]();
		for (let i = 0; i < movedXs.length; i++) if (movedXs[i] !== xs[i]) movedXs[i][Symbol.dispose]();
		const carryOut = results.slice(0, numCarry);
		const ysOut = results.slice(numCarry);
		const movedYs = ysOut.map((y) => {
			const moved = moveaxis(y, 1, 0);
			if (moved !== y) y[Symbol.dispose]();
			return moved;
		});
		return [[...carryOut, ...movedYs], require_backend.rep(numCarry + numY, 0)];
	}
};
const vmapJaxprCache = /* @__PURE__ */ new Map();
_registerJitCacheDisposer(() => {
	for (const inner$1 of vmapJaxprCache.values()) {
		for (const jaxpr of inner$1.values()) jaxpr.dispose();
		inner$1.clear();
	}
	vmapJaxprCache.clear();
});
function vmapJaxpr(jaxpr, axisSize, dims) {
	const cacheKey = JSON.stringify([axisSize, dims]);
	const prevResult = vmapJaxprCache.get(jaxpr)?.get(cacheKey);
	if (prevResult) return prevResult;
	const inAvals = jaxpr.inBinders.map((v, i) => {
		if (dims[i] === null) return v.aval;
		const shape$1 = [...v.aval.shape];
		shape$1.splice(dims[i], 0, axisSize);
		return new ShapedArray(shape$1, v.aval.dtype, v.aval.weakType);
	});
	const { jaxpr: newJaxpr } = makeJaxpr$1((args) => vmapFlat(jaxprAsFun(jaxpr), dims, args))(inAvals);
	if (!vmapJaxprCache.has(jaxpr)) vmapJaxprCache.set(jaxpr, /* @__PURE__ */ new Map());
	vmapJaxprCache.get(jaxpr).set(cacheKey, newJaxpr);
	return newJaxpr;
}
function vmapFlat(f, inAxes, args) {
	let axisSize = void 0;
	for (let i = 0; i < args.length; i++) if (inAxes[i] !== null) {
		const arg = args[i];
		if (!(arg instanceof Tracer)) throw new TypeError("vmap requires Tracer argument for mapped axes");
		const size$1 = arg.shape[inAxes[i]];
		if (axisSize === void 0) axisSize = size$1;
		else if (axisSize !== size$1) throw new TypeError("vmap requires all mapped axes to have the same size");
	}
	if (axisSize === void 0) throw new TypeError("vmap requires at least one mapped axis");
	let valsOut, bdimsOut;
	let shouldDisposeValOut = false;
	try {
		var _usingCtx$1 = (0, import_usingCtx$9.default)();
		const main = _usingCtx$1.u(newMain(BatchTrace, {
			axisSize,
			pureIntermediates: [],
			intermediates: []
		}));
		const trace$1 = new BatchTrace(main);
		const tracersIn = args.map((x, i) => inAxes[i] === null ? pureArray(x) : new BatchTracer(trace$1, pureArray(x), inAxes[i]));
		const outs = f(...tracersIn);
		const tracersOut = outs.map((out) => fullRaise(trace$1, out));
		[valsOut, bdimsOut] = require_backend.unzip2(tracersOut.map((t) => [t.val, t.batchDim]));
		shouldDisposeValOut = !hasAbstractTraceBelow(main.level);
		if (shouldDisposeValOut) {
			const outputTracers = new Set(tracersOut);
			const outputVals = new Set(valsOut);
			for (const t of trace$1.intermediates) {
				if (outputTracers.has(t)) continue;
				if (outputVals.has(t.val)) continue;
				try {
					t.dispose();
				} catch {}
			}
			const outputSet = new Set(valsOut);
			const disposed = /* @__PURE__ */ new Set();
			for (const arr of trace$1.pureIntermediates) {
				if (outputSet.has(arr) || disposed.has(arr)) continue;
				disposed.add(arr);
				if (arr.refCount > 0) arr[Symbol.dispose]?.();
			}
		}
	} catch (_) {
		_usingCtx$1.e = _;
	} finally {
		_usingCtx$1.d();
	}
	return require_backend.zip(valsOut, bdimsOut).map(([valOut, bdim]) => {
		const result = moveBatchAxis(axisSize, bdim, 0, valOut);
		if (result !== valOut) if (shouldDisposeValOut) try {
			valOut.dispose();
		} catch {}
		else valOut[Symbol.dispose]();
		return result;
	});
}
function vmap$1(f, inAxes = 0) {
	return (...args) => {
		const [argsFlat, inTree] = flatten(args);
		let inAxesFlat = [];
		if (typeof inAxes === "number") inAxesFlat = require_backend.rep(argsFlat.length, inAxes);
		else for (let i = 0; i < args.length; i++) if (inAxes[i] == null) inAxesFlat.push(...require_backend.rep(inTree.childTreedefs[i].size, null));
		else if (typeof inAxes[i] === "number") inAxesFlat.push(...require_backend.rep(inTree.childTreedefs[i].size, inAxes[i]));
		else {
			const [axesFlat, axesTreeDef] = flatten(inAxes[i]);
			if (!inTree.childTreedefs[i].equals(axesTreeDef)) throw new TreeMismatchError("vmap", inTree.childTreedefs[i], axesTreeDef);
			inAxesFlat.push(...axesFlat);
		}
		const [fFlat, outTree] = flattenFun(f, inTree);
		const outsFlat = vmapFlat(fFlat, inAxesFlat, argsFlat);
		if (outTree.value === void 0) throw new Error("outTree was not set in vmap");
		return unflatten(outTree.value, outsFlat);
	};
}
function jacfwd$1(f) {
	return function jacobianForward(x) {
		if (x.shape.length !== 1) throw new TypeError("jacfwd only supports 1D inputs");
		const [size$1] = x.shape;
		const pushfwd = (v) => {
			const [primals, tangents] = jvp$1(f, [x], [v]);
			if (!insideAbstractTrace()) {
				const [primalsFlat] = flatten(primals);
				for (const p of primalsFlat) if (p instanceof Tracer) try {
					p.dispose();
				} catch {}
			} else {
				const [primalsFlat] = flatten(primals);
				for (const p of primalsFlat) if (p instanceof Tracer) p[Symbol.dispose]();
			}
			return tangents;
		};
		const eyeMatrix = eye(size$1, void 0, { dtype: x.dtype });
		const result = vmap$1(pushfwd, [0])(eyeMatrix);
		eyeMatrix.dispose();
		return result;
	};
}

//#endregion
//#region src/frontend/jvp.ts
var import_usingCtx$8 = /* @__PURE__ */ __toESM(require_usingCtx(), 1);
var JVPTracer = class extends Tracer {
	#rc = 1;
	constructor(trace$1, primal, tangent) {
		super(trace$1);
		this.primal = primal;
		this.tangent = tangent;
	}
	get aval() {
		return this.primal.aval;
	}
	toString() {
		return `JVPTracer(${this.primal.toString()}, ${this.tangent.toString()})`;
	}
	get ref() {
		this.#rc++;
		return this;
	}
	dispose() {
		if (--this.#rc === 0) {
			this.primal.dispose();
			this.tangent.dispose();
		}
	}
	/**
	* Override Tracer's no-op to cascade disposal to primal and tangent.
	*
	* During PE tracing, JVPTracers may wrap concrete Arrays tracked in
	* PE's knownIntermediates. Disposing via `using` would conflict with
	* PE's lifecycle management, so we skip when inside PE scope.
	*
	* Also skip when a lower abstract trace exists. In nested abstract
	* compositions (e.g., makeJaxpr(jvp(...))), core.bind may call
	* Symbol.dispose on raised raw-literal arguments. Cascading here would free
	* primals/tangents that have already been captured as Jaxpr consts.
	*/
	[Symbol.dispose]() {
		if (!_peArrayCreationTracker && !hasAbstractTraceBelow(this._trace.main.level)) this.dispose();
	}
};
var JVPTrace = class extends Trace {
	pure(val) {
		return this.lift(pureArray(val));
	}
	lift(val) {
		const zero = zerosLike$1(val);
		if (zero instanceof Array$1) anonymousConstArrays.add(zero);
		const data = this.main.globalData;
		if (data) data.liftedTangents.push(zero);
		return new JVPTracer(this, val, zero);
	}
	processPrimitive(primitive, tracers, params) {
		const [primalsIn, tangentsIn] = require_backend.unzip2(tracers.map((x) => [x.primal, x.tangent]));
		const jvpRule = jvpRules[primitive];
		if (jvpRule === void 0) throw new Error(`No JVP rule for: ${primitive}`);
		const [primalsOut, tangentsOut] = jvpRule(primalsIn, tangentsIn, params);
		const result = require_backend.zip(primalsOut, tangentsOut).map(([x, t]) => new JVPTracer(this, x, t));
		const data = this.main.globalData;
		if (data) data.intermediates.push(...result);
		return result;
	}
};
/** Rule that applies the same operation to primals and tangents. */
function linearTangentsJvp(primitive) {
	return (primals, tangents, params) => {
		const ys = bind(primitive, primals, params);
		const dys = bind(primitive, tangents, params);
		return [ys, dys];
	};
}
/** Rule for product of gradients in bilinear operations. */
function bilinearTangentsJvp(primitive) {
	return ([x, y], [dx, dy], params) => {
		try {
			var _usingCtx$1 = (0, import_usingCtx$8.default)();
			const primal = bind1(primitive, [x, y], params);
			const xdy = _usingCtx$1.u(bind1(primitive, [x, dy], params));
			const dxy = _usingCtx$1.u(bind1(primitive, [dx, y], params));
			const tangent = xdy.add(dxy);
			return [[primal], [tangent]];
		} catch (_) {
			_usingCtx$1.e = _;
		} finally {
			_usingCtx$1.d();
		}
	};
}
/** Rule that zeros out any tangents. */
function zeroTangentsJvp(primitive) {
	return (primals, tangents, params) => {
		const ys = bind(primitive, primals, params);
		return [ys, ys.map((y) => zerosLike$1(y))];
	};
}
/** Compute `a @ b.T`, batched to last two axes. */
function batchMatmulT(a, b) {
	try {
		var _usingCtx3 = (0, import_usingCtx$8.default)();
		const aReshaped = _usingCtx3.u(a.reshape(a.shape.toSpliced(-1, 0, 1)));
		const bReshaped = _usingCtx3.u(b.reshape(b.shape.toSpliced(-2, 0, 1)));
		return dot$2(aReshaped, bReshaped);
	} catch (_) {
		_usingCtx3.e = _;
	} finally {
		_usingCtx3.d();
	}
}
/** Batch matrix transpose. */
function mT(a) {
	return moveaxis(a, -2, -1);
}
function sliceAxis(a, axis, p) {
	const slices = globalThis.Array(a.shape.length).fill([]);
	slices[require_backend.checkAxis(axis, a.ndim)] = p;
	return a.slice(...slices);
}
function padAxis(a, axis, p) {
	const pads = globalThis.Array(a.shape.length).fill([0, 0]);
	pads[require_backend.checkAxis(axis, a.ndim)] = p;
	return pad$1(a, pads);
}
const jvpRules = {
	[Primitive.Add]: linearTangentsJvp(Primitive.Add),
	[Primitive.Mul]: bilinearTangentsJvp(Primitive.Mul),
	[Primitive.Idiv]: zeroTangentsJvp(Primitive.Idiv),
	[Primitive.Mod]([x, y], [dx, dy]) {
		try {
			var _usingCtx4 = (0, import_usingCtx$8.default)();
			if (!require_backend.isFloatDtype(x.dtype) && !require_backend.isFloatDtype(y.dtype)) {
				const result = mod(x, y);
				return [[result], [zerosLike$1(result)]];
			}
			const q = _usingCtx4.u(idiv(x, y));
			const dyq = _usingCtx4.u(dy.mul(q));
			const tangent = dx.sub(dyq);
			return [[mod(x, y)], [tangent]];
		} catch (_) {
			_usingCtx4.e = _;
		} finally {
			_usingCtx4.d();
		}
	},
	[Primitive.Min]([x, y], [dx, dy]) {
		try {
			var _usingCtx5 = (0, import_usingCtx$8.default)();
			const cond = _usingCtx5.u(less$1(y, x));
			return [[min$1(x, y)], [where$1(cond, dy, dx)]];
		} catch (_) {
			_usingCtx5.e = _;
		} finally {
			_usingCtx5.d();
		}
	},
	[Primitive.Max]([x, y], [dx, dy]) {
		try {
			var _usingCtx6 = (0, import_usingCtx$8.default)();
			const cond = _usingCtx6.u(less$1(x, y));
			return [[max$1(x, y)], [where$1(cond, dy, dx)]];
		} catch (_) {
			_usingCtx6.e = _;
		} finally {
			_usingCtx6.d();
		}
	},
	[Primitive.Neg]: linearTangentsJvp(Primitive.Neg),
	[Primitive.Reciprocal]([x], [dx]) {
		try {
			var _usingCtx7 = (0, import_usingCtx$8.default)();
			const xRecip = reciprocal$1(x);
			const xRecipSq = _usingCtx7.u(xRecip.mul(xRecip));
			const negXRecipSq = _usingCtx7.u(neg(xRecipSq));
			return [[xRecip], [negXRecipSq.mul(dx)]];
		} catch (_) {
			_usingCtx7.e = _;
		} finally {
			_usingCtx7.d();
		}
	},
	[Primitive.Floor]: zeroTangentsJvp(Primitive.Floor),
	[Primitive.Ceil]: zeroTangentsJvp(Primitive.Ceil),
	[Primitive.StopGradient]: zeroTangentsJvp(Primitive.StopGradient),
	[Primitive.Cast]([x], [dx], { dtype }) {
		if (x.dtype === dtype) return [[x], [dx]];
		if (require_backend.isFloatDtype(dtype) && require_backend.isFloatDtype(x.dtype)) return [[cast(x, dtype)], [cast(dx, dtype)]];
		else return [[cast(x, dtype)], [zerosLike$1(x)]];
	},
	[Primitive.Bitcast]([x], [dx], { dtype }) {
		if (x.dtype === dtype) return [[x], [dx]];
		return [[bitcast(x, dtype)], [zerosLike$1(x)]];
	},
	[Primitive.Sin]([x], [dx]) {
		try {
			var _usingCtx8 = (0, import_usingCtx$8.default)();
			const cosX = _usingCtx8.u(cos$1(x));
			return [[sin$1(x)], [cosX.mul(dx)]];
		} catch (_) {
			_usingCtx8.e = _;
		} finally {
			_usingCtx8.d();
		}
	},
	[Primitive.Cos]([x], [dx]) {
		try {
			var _usingCtx9 = (0, import_usingCtx$8.default)();
			const sinX = _usingCtx9.u(sin$1(x));
			const negSinX = _usingCtx9.u(neg(sinX));
			return [[cos$1(x)], [negSinX.mul(dx)]];
		} catch (_) {
			_usingCtx9.e = _;
		} finally {
			_usingCtx9.d();
		}
	},
	[Primitive.Asin]([x], [dx]) {
		try {
			var _usingCtx10 = (0, import_usingCtx$8.default)();
			const one = _usingCtx10.u(cast(1, x.dtype));
			const xSq = _usingCtx10.u(x.mul(x));
			const oneMinusXSq = _usingCtx10.u(one.sub(xSq));
			const recip = _usingCtx10.u(reciprocal$1(oneMinusXSq));
			const denom = _usingCtx10.u(sqrt$1(recip));
			return [[asin$1(x)], [denom.mul(dx)]];
		} catch (_) {
			_usingCtx10.e = _;
		} finally {
			_usingCtx10.d();
		}
	},
	[Primitive.Atan]([x], [dx]) {
		try {
			var _usingCtx11 = (0, import_usingCtx$8.default)();
			const one = _usingCtx11.u(cast(1, x.dtype));
			const xSq = _usingCtx11.u(x.mul(x));
			const denom = _usingCtx11.u(one.add(xSq));
			return [[atan$1(x)], [dx.div(denom)]];
		} catch (_) {
			_usingCtx11.e = _;
		} finally {
			_usingCtx11.d();
		}
	},
	[Primitive.Exp]([x], [dx]) {
		const z = exp$1(x);
		return [[z], [z.mul(dx)]];
	},
	[Primitive.Log]([x], [dx]) {
		try {
			var _usingCtx12 = (0, import_usingCtx$8.default)();
			const recipX = _usingCtx12.u(reciprocal$1(x));
			return [[log$1(x)], [recipX.mul(dx)]];
		} catch (_) {
			_usingCtx12.e = _;
		} finally {
			_usingCtx12.d();
		}
	},
	[Primitive.Erf]([x], [dx]) {
		try {
			var _usingCtx13 = (0, import_usingCtx$8.default)();
			const coeff = 2 / Math.sqrt(Math.PI);
			const xSq = _usingCtx13.u(x.mul(x));
			const negXSq = _usingCtx13.u(neg(xSq));
			const expTerm = _usingCtx13.u(exp$1(negXSq));
			const scaled = _usingCtx13.u(expTerm.mul(coeff));
			return [[erf$1(x)], [scaled.mul(dx)]];
		} catch (_) {
			_usingCtx13.e = _;
		} finally {
			_usingCtx13.d();
		}
	},
	[Primitive.Erfc]([x], [dx]) {
		try {
			var _usingCtx14 = (0, import_usingCtx$8.default)();
			const coeff = -2 / Math.sqrt(Math.PI);
			const xSq = _usingCtx14.u(x.mul(x));
			const negXSq = _usingCtx14.u(neg(xSq));
			const expTerm = _usingCtx14.u(exp$1(negXSq));
			const scaled = _usingCtx14.u(expTerm.mul(coeff));
			return [[erfc$1(x)], [scaled.mul(dx)]];
		} catch (_) {
			_usingCtx14.e = _;
		} finally {
			_usingCtx14.d();
		}
	},
	[Primitive.Sqrt]([x], [dx]) {
		try {
			var _usingCtx15 = (0, import_usingCtx$8.default)();
			const z = sqrt$1(x);
			const z2 = _usingCtx15.u(z.mul(2));
			const recipZ2 = _usingCtx15.u(reciprocal$1(z2));
			return [[z], [recipZ2.mul(dx)]];
		} catch (_) {
			_usingCtx15.e = _;
		} finally {
			_usingCtx15.d();
		}
	},
	[Primitive.Reduce]([x], [dx], { op, axis }) {
		if (op === require_backend.AluOp.Add) return [[reduce(x, op, axis)], [reduce(dx, op, axis)]];
		else if (op === require_backend.AluOp.Mul) try {
			var _usingCtx16 = (0, import_usingCtx$8.default)();
			const primal = reduce(x, op, axis);
			const bcast = _usingCtx16.u(broadcast(primal, x.shape, axis));
			const recip = _usingCtx16.u(reciprocal$1(x));
			const bcastTimesRecip = _usingCtx16.u(bcast.mul(recip));
			const bcastTimesRecipTimesDx = _usingCtx16.u(bcastTimesRecip.mul(dx));
			const tangent = bcastTimesRecipTimesDx.sum(axis);
			return [[primal], [tangent]];
		} catch (_) {
			_usingCtx16.e = _;
		} finally {
			_usingCtx16.d();
		}
		else if (op === require_backend.AluOp.Min || op === require_backend.AluOp.Max) try {
			var _usingCtx17 = (0, import_usingCtx$8.default)();
			const primal = reduce(x, op, axis);
			const bcastPrimal = _usingCtx17.u(broadcast(primal, x.shape, axis));
			const notMin = _usingCtx17.u(notEqual$1(x, bcastPrimal));
			const whereNotMin0 = _usingCtx17.u(where$1(notMin, 0, 1));
			const minCount = _usingCtx17.u(whereNotMin0.sum(axis));
			const whereNotMin0Dx = _usingCtx17.u(where$1(notMin, 0, dx));
			const sumDx = _usingCtx17.u(whereNotMin0Dx.sum(axis));
			const tangent = sumDx.div(minCount);
			return [[primal], [tangent]];
		} catch (_) {
			_usingCtx17.e = _;
		} finally {
			_usingCtx17.d();
		}
		else throw new Error(`JVP rule not implemented for reduce op: ${op}`);
	},
	[Primitive.Pool]: linearTangentsJvp(Primitive.Pool),
	[Primitive.PoolTranspose]: linearTangentsJvp(Primitive.PoolTranspose),
	[Primitive.Dot]: bilinearTangentsJvp(Primitive.Dot),
	[Primitive.Conv]: bilinearTangentsJvp(Primitive.Conv),
	[Primitive.Compare]: zeroTangentsJvp(Primitive.Compare),
	[Primitive.Where]([cond, x, y], [dcond, dx, dy]) {
		return [[where$1(cond, x, y)], [where$1(cond, dx, dy)]];
	},
	[Primitive.Concatenate]: linearTangentsJvp(Primitive.Concatenate),
	[Primitive.Split]: linearTangentsJvp(Primitive.Split),
	[Primitive.RandomBits]: zeroTangentsJvp(Primitive.RandomBits),
	[Primitive.Gather]([x, ...indices], [dx, ..._], { axis, outDim }) {
		const indicesRef = indices;
		return [[gather(x, indices, axis, outDim)], [gather(dx, indicesRef, axis, outDim)]];
	},
	[Primitive.Transpose]: linearTangentsJvp(Primitive.Transpose),
	[Primitive.Broadcast]: linearTangentsJvp(Primitive.Broadcast),
	[Primitive.Reshape]: linearTangentsJvp(Primitive.Reshape),
	[Primitive.Flip]: linearTangentsJvp(Primitive.Flip),
	[Primitive.Shrink]: linearTangentsJvp(Primitive.Shrink),
	[Primitive.Pad]: linearTangentsJvp(Primitive.Pad),
	[Primitive.Sort]([x], [dx]) {
		const [y, idx] = argsort$1(x);
		const gatherResult = gather(dx, [idx], [-1], -1);
		if (!_peArrayCreationTracker) idx.dispose();
		return [[y], [gatherResult]];
	},
	[Primitive.Argsort]([x], [dx]) {
		const [y, idx] = argsort$1(x);
		const gatherResult = gather(dx, [idx], [-1], -1);
		const zerosIdx = zerosLike$1(idx);
		return [[y, idx], [gatherResult, zerosIdx]];
	},
	[Primitive.TriangularSolve]([a, b], [da, db], { unitDiagonal }) {
		try {
			var _usingCtx18 = (0, import_usingCtx$8.default)();
			const x = triangularSolve$1(a, b, { unitDiagonal });
			const dax = _usingCtx18.u(batchMatmulT(da, x));
			const mTdax = _usingCtx18.u(mT(dax));
			const rhsT = _usingCtx18.u(db.sub(mTdax));
			const dx = triangularSolve$1(a, rhsT, { unitDiagonal });
			return [[x], [dx]];
		} catch (_) {
			_usingCtx18.e = _;
		} finally {
			_usingCtx18.d();
		}
	},
	[Primitive.Cholesky]([a], [da]) {
		try {
			var _usingCtx19 = (0, import_usingCtx$8.default)();
			const L = cholesky$2(a);
			const mTda = _usingCtx19.u(mT(da));
			const daSymm = _usingCtx19.u(da.add(mTda));
			const da2 = _usingCtx19.u(daSymm.mul(.5));
			const W = _usingCtx19.u(triangularSolve$1(L, da2, { lower: true }));
			const mTW = _usingCtx19.u(mT(W));
			const ST = _usingCtx19.u(triangularSolve$1(L, mTW, { lower: true }));
			const triuST1 = _usingCtx19.u(triu(ST, 1));
			const triuST0 = _usingCtx19.u(triu(ST));
			const triuSum = _usingCtx19.u(triuST1.add(triuST0));
			const triuHalf = _usingCtx19.u(triuSum.mul(.5));
			const dL = batchMatmulT(L, triuHalf);
			return [[L], [dL]];
		} catch (_) {
			_usingCtx19.e = _;
		} finally {
			_usingCtx19.d();
		}
	},
	[Primitive.LU]([a], [da]) {
		try {
			var _usingCtx20 = (0, import_usingCtx$8.default)();
			const [luMatrix, pivots, permutation] = lu$1(a);
			const [m, n] = a.shape.slice(-2);
			const k = Math.min(m, n);
			const luSliceL = _usingCtx20.u(sliceAxis(luMatrix, -1, [0, k]));
			const lLower = tril(luSliceL, -1);
			const lPaddedNeedsDispose = m > k;
			const lPadded = lPaddedNeedsDispose ? padAxis(lLower, -1, [0, m - k]) : lLower;
			const eyeM = _usingCtx20.u(eye(m));
			const L = _usingCtx20.u(lPadded.add(eyeM));
			if (lPaddedNeedsDispose) lPadded[Symbol.dispose]();
			lLower[Symbol.dispose]();
			const luSliceU = _usingCtx20.u(sliceAxis(luMatrix, -2, [0, k]));
			const uUpper = triu(luSliceU);
			const uPaddedNeedsDispose = n > k;
			const uPadded = uPaddedNeedsDispose ? padAxis(uUpper, -2, [0, n - k]) : uUpper;
			const uEye = _usingCtx20.u(n > k ? (() => {
				try {
					var _usingCtx21 = (0, import_usingCtx$8.default)();
					const innerEye = _usingCtx21.u(eye(n - k));
					const padded1 = _usingCtx21.u(padAxis(innerEye, -1, [k, 0]));
					return padAxis(padded1, -2, [k, 0]);
				} catch (_) {
					_usingCtx21.e = _;
				} finally {
					_usingCtx21.d();
				}
			})() : zerosLike$1(uPadded));
			const U = _usingCtx20.u(uPadded.add(uEye));
			if (uPaddedNeedsDispose) uPadded[Symbol.dispose]();
			uUpper[Symbol.dispose]();
			const permReshaped = _usingCtx20.u(permutation.reshape([...permutation.shape, 1]));
			const arangeM = _usingCtx20.u(arange(m));
			const permEq = _usingCtx20.u(permReshaped.equal(arangeM));
			const P = _usingCtx20.u(permEq.astype(da.dtype));
			const mTda = _usingCtx20.u(mT(da));
			const pda = _usingCtx20.u(batchMatmulT(P, mTda));
			const mTpda = _usingCtx20.u(mT(pda));
			const solvedPda = _usingCtx20.u(triangularSolve$1(L, mTpda, {
				lower: true,
				unitDiagonal: true
			}));
			const la = _usingCtx20.u(mT(solvedPda));
			const mTU = _usingCtx20.u(mT(U));
			const lau = _usingCtx20.u(triangularSolve$1(mTU, la, { lower: true }));
			const trilLau = _usingCtx20.u(tril(lau, -1));
			const mTtrilLau = _usingCtx20.u(mT(trilLau));
			const lDot = _usingCtx20.u(batchMatmulT(L, mTtrilLau));
			const triuLau = _usingCtx20.u(triu(lau));
			const mTU2 = _usingCtx20.u(mT(U));
			const uDot = _usingCtx20.u(batchMatmulT(triuLau, mTU2));
			const luDot = lDot.add(uDot);
			const zerosPivots = zerosLike$1(pivots);
			const zerosPerm = zerosLike$1(permutation);
			return [[
				luMatrix,
				pivots,
				permutation
			], [
				luDot,
				zerosPivots,
				zerosPerm
			]];
		} catch (_) {
			_usingCtx20.e = _;
		} finally {
			_usingCtx20.d();
		}
	},
	[Primitive.Jit](primals, tangents, { name, jaxpr }) {
		const newJaxpr = jvpJaxpr(jaxpr);
		const outs = bind(Primitive.Jit, [
			...newJaxpr.consts,
			...primals,
			...tangents
		], {
			name: `${name}_jvp`,
			jaxpr: newJaxpr.jaxpr,
			numConsts: newJaxpr.consts.length
		});
		const n = outs.length / 2;
		if (!Number.isInteger(n)) throw new Error("internal: JVP Jaxpr output length is not even");
		const [primalsOut, tangentsOut] = [outs.slice(0, n), outs.slice(n)];
		return [primalsOut, tangentsOut];
	},
	[Primitive.DynamicUpdateSlice]([dst, src], [ddst, dsrc], { offset, axis }) {
		throw new Error("JVP: dynamic_update_slice is not implemented");
	},
	[Primitive.Scan](primals, tangents, { jaxpr, numCarry, numConsts, length, reverse, checkpoint }) {
		const numX = primals.length - numConsts - numCarry;
		const numY = jaxpr.outs.length - numCarry;
		const jvpBody = jvpJaxpr(jaxpr);
		const numJvpConsts = jvpBody.consts.length;
		const numBodyInputs = numConsts + numCarry + numX;
		const jvpOrderAvals = jvpBody.jaxpr.inBinders.slice(numJvpConsts).map((v) => v.aval);
		const constsP_avals = jvpOrderAvals.slice(0, numConsts);
		const carryP_avals = jvpOrderAvals.slice(numConsts, numConsts + numCarry);
		const xP_avals = jvpOrderAvals.slice(numConsts + numCarry, numBodyInputs);
		const constsT_avals = jvpOrderAvals.slice(numBodyInputs, numBodyInputs + numConsts);
		const carryT_avals = jvpOrderAvals.slice(numBodyInputs + numConsts, numBodyInputs + numConsts + numCarry);
		const xT_avals = jvpOrderAvals.slice(numBodyInputs + numConsts + numCarry);
		const wrapperInAvals = [
			...constsP_avals,
			...constsT_avals,
			...carryP_avals,
			...carryT_avals,
			...xP_avals,
			...xT_avals
		];
		const { jaxpr: wrapperJaxpr } = makeJaxpr$1((...scanOrderArgs) => {
			const constsP_in = scanOrderArgs.slice(0, numConsts);
			const constsT_in = scanOrderArgs.slice(numConsts, numConsts * 2);
			const carryP_in = scanOrderArgs.slice(numConsts * 2, numConsts * 2 + numCarry);
			const carryT_in = scanOrderArgs.slice(numConsts * 2 + numCarry, numConsts * 2 + numCarry * 2);
			const xP_in = scanOrderArgs.slice(numConsts * 2 + numCarry * 2, numConsts * 2 + numCarry * 2 + numX);
			const xT_in = scanOrderArgs.slice(numConsts * 2 + numCarry * 2 + numX);
			const jvpOrderArgs = [
				...constsP_in,
				...carryP_in,
				...xP_in,
				...constsT_in,
				...carryT_in,
				...xT_in
			];
			const jvpOutputs = bind(Primitive.Jit, [...jvpBody.consts, ...jvpOrderArgs], {
				jaxpr: jvpBody.jaxpr,
				numConsts: numJvpConsts,
				name: "jvp_body"
			});
			const carryP_out = jvpOutputs.slice(0, numCarry);
			const yP_out = jvpOutputs.slice(numCarry, numCarry + numY);
			const carryT_out = jvpOutputs.slice(numCarry + numY, numCarry * 2 + numY);
			const yT_out = jvpOutputs.slice(numCarry * 2 + numY);
			return [
				...carryP_out,
				...carryT_out,
				...yP_out,
				...yT_out
			];
		})(...wrapperInAvals);
		const constsP = primals.slice(0, numConsts);
		const carryP = primals.slice(numConsts, numConsts + numCarry);
		const xsP = primals.slice(numConsts + numCarry);
		const constsT = tangents.slice(0, numConsts);
		const carryT = tangents.slice(numConsts, numConsts + numCarry);
		const xsT = tangents.slice(numConsts + numCarry);
		const scanArgsJvp = [
			...wrapperJaxpr.consts,
			...constsP,
			...constsT,
			...carryP,
			...carryT,
			...xsP,
			...xsT
		];
		const results = bind(Primitive.Scan, scanArgsJvp, {
			jaxpr: wrapperJaxpr.jaxpr,
			numCarry: numCarry * 2,
			numConsts: wrapperJaxpr.consts.length + numConsts * 2,
			length,
			reverse,
			checkpoint
		});
		wrapperJaxpr.dispose();
		const carryOutP = results.slice(0, numCarry);
		const carryOutT = results.slice(numCarry, numCarry * 2);
		const ysP = results.slice(numCarry * 2, numCarry * 2 + numY);
		const ysT = results.slice(numCarry * 2 + numY);
		const primalsOut = [...carryOutP, ...ysP];
		const tangentsOut = [...carryOutT, ...ysT];
		return [primalsOut, tangentsOut];
	}
};
const jvpJaxprCache = /* @__PURE__ */ new Map();
_registerJitCacheDisposer(() => {
	for (const cj of jvpJaxprCache.values()) cj.dispose();
	jvpJaxprCache.clear();
});
function jvpJaxpr(jaxpr) {
	if (jvpJaxprCache.has(jaxpr)) return jvpJaxprCache.get(jaxpr);
	const inAvals = jaxpr.inBinders.map((v) => v.aval);
	const { jaxpr: newJaxpr } = makeJaxpr$1((primals, tangents) => jvpFlat(jaxprAsFun(jaxpr), primals, tangents))(inAvals, inAvals);
	jvpJaxprCache.set(jaxpr, newJaxpr);
	return newJaxpr;
}
function jvpFlat(f, primals, tangents) {
	try {
		var _usingCtx22 = (0, import_usingCtx$8.default)();
		const jvpData = {
			intermediates: [],
			liftedTangents: []
		};
		const main = _usingCtx22.u(newMain(JVPTrace, jvpData));
		main.isAbstract = true;
		const trace$1 = new JVPTrace(main);
		const newlyCreatedInputs = [];
		const tracersIn = require_backend.zip(primals, tangents).map(([x, t]) => {
			const px = pureArray(x);
			const pt = pureArray(t);
			if (px !== x) newlyCreatedInputs.push(px);
			if (pt !== t) newlyCreatedInputs.push(pt);
			return new JVPTracer(trace$1, px, pt);
		});
		const outs = f(...tracersIn);
		const tracersOut = outs.map((out) => fullRaise(trace$1, out));
		const result = require_backend.unzip2(tracersOut.map((t) => [t.primal, t.tangent]));
		if (!hasAbstractTraceBelow(main.level)) {
			const outputSet = new Set(tracersOut);
			for (const t of jvpData.intermediates) if (!outputSet.has(t)) {
				if (t.primal.refCount > 0) t.primal[Symbol.dispose]();
				if (t.tangent.refCount > 0) t.tangent[Symbol.dispose]();
			}
			const outputArrays = new Set([...result[0], ...result[1]]);
			for (const a of newlyCreatedInputs) if (!outputArrays.has(a) && a.refCount > 0) a[Symbol.dispose]?.();
			const outputTangents = new Set(result[1]);
			for (const z of jvpData.liftedTangents) if (!outputTangents.has(z) && z.refCount > 0) z[Symbol.dispose]?.();
		}
		return result;
	} catch (_) {
		_usingCtx22.e = _;
	} finally {
		_usingCtx22.d();
	}
}
function jvp$1(f, primals, tangents, { hasAux = false } = {}) {
	const [primalsFlat, inTree] = flatten(primals);
	const [tangentsFlat, inTree2] = flatten(tangents);
	if (!inTree.equals(inTree2)) throw new TreeMismatchError("jvp", inTree, inTree2);
	let flatFun, outTree, aux;
	if (hasAux) [flatFun, outTree, aux] = flattenFunWithAux(f, inTree);
	else [flatFun, outTree] = flattenFun(f, inTree);
	const [primalsOutFlat, tangentsOutFlat] = jvpFlat(flatFun, primalsFlat, tangentsFlat);
	if (outTree.value === void 0) throw new Error("outTree was not set in jvp");
	const primalsOut = unflatten(outTree.value, primalsOutFlat);
	const tangentsOut = unflatten(outTree.value, tangentsOutFlat);
	if (hasAux) return [
		primalsOut,
		tangentsOut,
		lowerAux(aux.value)
	];
	return [primalsOut, tangentsOut];
}
/** Lowering for auxiliary data returned in `hasAux: true` methods. */
function lowerAux(aux) {
	const level = currentTraceLevel();
	return map((x) => {
		if (x instanceof Tracer) while (x._trace.main.level > level) if (x instanceof JVPTracer) x = x.primal;
		else {
			const y = x.fullLower();
			if (y._trace.main.level >= x._trace.main.level) throw new Error("internal: lowerAux did not reduce trace level");
			x = y;
		}
		return x;
	}, aux);
}

//#endregion
//#region src/frontend/linearize.ts
var import_usingCtx$7 = /* @__PURE__ */ __toESM(require_usingCtx(), 1);
/** Array value that can either be known or unknown. */
var PartialVal = class PartialVal {
	constructor(val, aval) {
		this.val = val;
		this.aval = aval;
	}
	static known(val) {
		return new PartialVal(val, ShapedArray.fromAval(val.aval));
	}
	static unknown(aval) {
		return new PartialVal(null, ShapedArray.fromAval(aval));
	}
	get isKnown() {
		return this.val !== null;
	}
	toString() {
		return this.val ? this.val.toString() : this.aval.toString();
	}
};
function partialEvalFlat(f, pvalsIn) {
	let jaxpr;
	let pvalsOut;
	let knownIntermediates;
	let literalIntermediates;
	try {
		var _usingCtx$1 = (0, import_usingCtx$7.default)();
		const main = _usingCtx$1.u(newMain(PartialEvalTrace));
		main.isAbstract = true;
		const trace$1 = new PartialEvalTrace(main);
		const tracersIn = pvalsIn.map((pval) => trace$1.newArg(pval));
		const unknownTracersIn = tracersIn.filter((t) => !t.pval.isKnown).map((t) => t.ref);
		const previousTracker = _peArrayCreationTracker;
		const peCreatedArrays = [];
		_setPACT(peCreatedArrays);
		let outs;
		try {
			outs = f(...tracersIn);
		} catch (e$1) {
			_setPACT(previousTracker);
			for (const t of peCreatedArrays) try {
				t.dispose();
			} catch {}
			throw e$1;
		}
		_setPACT(previousTracker);
		const tracersOut = outs.map((out) => fullRaise(trace$1, out));
		pvalsOut = tracersOut.map((t) => t.pval);
		const unknownTracersOut = tracersOut.filter((t) => !t.pval.isKnown);
		jaxpr = partialEvalGraphToJaxpr(unknownTracersIn, unknownTracersOut);
		for (const ct of trace$1.allConstPETracers) {
			if (!ct.isAlive) continue;
			try {
				ct.dispose();
			} catch {}
		}
		knownIntermediates = trace$1.knownIntermediates;
		literalIntermediates = trace$1.literalIntermediates;
		knownIntermediates.push(...peCreatedArrays);
	} catch (_) {
		_usingCtx$1.e = _;
	} finally {
		_usingCtx$1.d();
	}
	return {
		jaxpr,
		pvalsOut,
		peIntermediates: knownIntermediates,
		literalIntermediates
	};
}
/**
* Unwrap tracer wrappers (BatchTracer, etc.) to find the underlying concrete
* Array. Chases the `.val` chain until it reaches a non-wrapper value.
* Returns the input unchanged if it's already a concrete Array.
*/
function unwrapToConcreteArray(t) {
	let current = t;
	while (current && typeof current === "object" && "val" in current) {
		const inner$1 = current.val;
		if (inner$1 === current || !(inner$1 instanceof Tracer)) break;
		current = inner$1;
	}
	return current;
}
/**
* Dispose PE intermediates that aren't outputs or externally captured.
* During PE, all-known evaluations create concrete arrays. Output values
* (in protectedVals) are returned to the caller. ClosedJaxpr consts have
* independent ownership via .ref in partialEvalGraphToJaxpr. Everything
* else (pure intermediates AND computed consts) needs one dispose here
* to balance the rc=1 from creation.
*/
function disposePeIntermediates(peIntermediates, _literalIntermediates, protectedVals) {
	if (insideAbstractTrace()) return;
	const allProtected = new Set(protectedVals);
	for (const v of protectedVals) {
		const concrete = unwrapToConcreteArray(v);
		if (concrete !== v) allProtected.add(concrete);
	}
	const targets = peIntermediates;
	const disposed = /* @__PURE__ */ new Set();
	for (const t of targets) {
		if (allProtected.has(t)) continue;
		if (disposed.has(t)) continue;
		const concrete = unwrapToConcreteArray(t);
		if (concrete !== t && allProtected.has(concrete)) continue;
		disposed.add(t);
		if (concrete !== t) disposed.add(concrete);
		try {
			t.dispose();
		} catch {}
	}
}
/**
* Extract concrete Arrays from a value tree that may contain JVPTracers,
* PartialEvalTracers, or other wrappers. Uses lowerAux to walk through
* the tracer chain (JVPTracer→PETracer→concrete Array).
*/
function collectConcreteArrays(value) {
	const result = [];
	const lowered = lowerAux(value);
	map((x) => {
		if (x instanceof Tracer) result.push(x);
		return x;
	}, lowered);
	return result;
}
/**
* Helper function with shared Jaxpr logic between linearize and vjp.
*
* Internally, vjp() looks very similar to linearize() but returns a function
* evaluating the "transposed" linearized Jaxpr, pulling back cotangents instead
* of pushing forward tangents.
*/
function linearizeFlatUtil(f, primalsIn) {
	const pvalsIn = [...primalsIn.map(PartialVal.known), ...primalsIn.map((t) => PartialVal.unknown(t.aval))];
	const fJvp = (...x) => {
		const k = x.length / 2;
		const [primalsOut$1, tangentsOut] = jvp$1(f, x.slice(0, k), x.slice(k, 2 * k));
		return [...primalsOut$1, ...tangentsOut];
	};
	const { jaxpr, pvalsOut, peIntermediates, literalIntermediates } = partialEvalFlat(fJvp, pvalsIn);
	const primalPvals = pvalsOut.slice(0, pvalsOut.length / 2);
	if (!primalPvals.every((pval) => pval.isKnown)) throw new Error("Not all primal values are known after partial evaluation");
	const primalsOut = primalPvals.map((pval) => pval.val);
	return {
		primalsOut,
		jaxpr,
		peIntermediates,
		literalIntermediates
	};
}
function linearizeFlat(f, primalsIn, auxStore) {
	const { primalsOut, jaxpr, peIntermediates, literalIntermediates } = linearizeFlatUtil(f, primalsIn);
	const protectedVals = new Set(primalsOut);
	for (const c of jaxpr.consts) if (c.refCount <= 1) protectedVals.add(c);
	if (auxStore?.value != null) for (const arr of collectConcreteArrays(auxStore.value)) protectedVals.add(arr);
	disposePeIntermediates(peIntermediates, literalIntermediates, protectedVals);
	const fLin = (...tangents) => evalJaxpr(jaxpr.jaxpr, [...jaxpr.consts.map((c) => c.ref), ...tangents]);
	const dispose$1 = () => jaxpr.dispose();
	return [
		primalsOut,
		fLin,
		dispose$1
	];
}
function linearize$1(f, primalsIn, { hasAux = false } = {}) {
	const [primalsInFlat, inTree] = flatten(primalsIn);
	let fFlat, outTree, aux;
	if (hasAux) [fFlat, outTree, aux] = flattenFunWithAux(f, inTree);
	else [fFlat, outTree] = flattenFun(f, inTree);
	const wrappedPrimals = primalsInFlat.map(pureArray);
	const [primalsOutFlat, fLinFlat, dispose$1] = linearizeFlat(fFlat, wrappedPrimals, hasAux ? aux : void 0);
	if (!insideAbstractTrace()) {
		const primalsOutSet = new Set(primalsOutFlat);
		for (let i = 0; i < wrappedPrimals.length; i++) if (wrappedPrimals[i] !== primalsInFlat[i] && !primalsOutSet.has(wrappedPrimals[i])) wrappedPrimals[i].dispose();
	}
	if (outTree.value === void 0) throw new Error("outTree was not set in linearize");
	const primalsOut = unflatten(outTree.value, primalsOutFlat);
	const fLin = ((...tangentsIn) => {
		const [tangentsInFlat, inTree2] = flatten(tangentsIn);
		if (!inTree.equals(inTree2)) throw new TreeMismatchError("linearize", inTree, inTree2);
		const tangentsOutFlat = fLinFlat(...tangentsInFlat.map(pureArray));
		return unflatten(outTree.value, tangentsOutFlat);
	});
	fLin.dispose = dispose$1;
	fLin[Symbol.dispose] = dispose$1;
	if (hasAux) return [
		primalsOut,
		fLin,
		lowerAux(aux.value)
	];
	return [primalsOut, fLin];
}
var PartialEvalTracer = class extends Tracer {
	#rc;
	constructor(trace$1, pval, recipe) {
		super(trace$1);
		this.pval = pval;
		this.recipe = recipe;
		this.#rc = 1;
	}
	get aval() {
		return this.pval.aval;
	}
	toString() {
		if (!this.recipe) return `PartialEvalTracer(${this.pval.toString()})`;
		else return `PartialEvalTracer<${this.recipe.type}>(${this.pval.toString()})`;
	}
	get ref() {
		if (this.#rc <= 0) throw new UseAfterFreeError(this);
		this.#rc++;
		return this;
	}
	/** Whether this PETracer hasn't been fully disposed yet. */
	get isAlive() {
		return this.#rc > 0;
	}
	dispose() {
		if (this.#rc <= 0) throw new UseAfterFreeError(this);
		if (--this.#rc === 0) {
			if (this.pval.isKnown) this.pval.val.dispose();
			else if (this.recipe?.type === "Const") this.recipe.val.dispose();
		}
	}
	fullLower() {
		if (this.pval.isKnown) return this.pval.val;
		return this;
	}
};
var PartialEvalTrace = class extends Trace {
	newArg(pval) {
		if (pval.isKnown) return new PartialEvalTracer(this, pval, null);
		return new PartialEvalTracer(this, pval, { type: "LambdaBinding" });
	}
	pure(val) {
		const arr = pureArray(val);
		if (!(val instanceof Tracer)) {
			this.knownIntermediates.push(arr);
			this.literalIntermediates.push(arr);
		}
		return new PartialEvalTracer(this, PartialVal.known(arr), null);
	}
	lift = this.pure;
	get knownIntermediates() {
		let arr = this.main.globalData?._knownIntermediates;
		if (!arr) {
			arr = [];
			if (!this.main.globalData) this.main.globalData = {};
			this.main.globalData._knownIntermediates = arr;
		}
		return arr;
	}
	get literalIntermediates() {
		let arr = this.main.globalData?._literalIntermediates;
		if (!arr) {
			arr = [];
			if (!this.main.globalData) this.main.globalData = {};
			this.main.globalData._literalIntermediates = arr;
		}
		return arr;
	}
	get allConstPETracers() {
		let arr = this.main.globalData?._allConstPETracers;
		if (!arr) {
			arr = [];
			if (!this.main.globalData) this.main.globalData = {};
			this.main.globalData._allConstPETracers = arr;
		}
		return arr;
	}
	instantiateConst(tracer) {
		if (!tracer.pval.isKnown) return tracer;
		else {
			const pval = PartialVal.unknown(ShapedArray.fromAval(tracer.aval));
			const val = tracer.pval.val.ref;
			const constTracer = new PartialEvalTracer(this, pval, {
				type: "Const",
				val
			});
			this.allConstPETracers.push(constTracer);
			return constTracer;
		}
	}
	processPrimitive(primitive, tracers, params) {
		if (tracers.every((t) => t.pval.isKnown)) {
			const results = bind(primitive, tracers.map((t) => t.fullLower()), params);
			for (const r of results) this.knownIntermediates.push(r);
			return results;
		}
		if (primitive === Primitive.Jit) {
			const { name, jaxpr, numConsts } = params;
			return this.#partialEvalJaxpr(name, jaxpr, numConsts, tracers);
		}
		if (primitive === Primitive.Scan) return this.#partialEvalScan(params, tracers);
		const tracersIn = tracers.map((t) => this.instantiateConst(t));
		const avalsIn = tracersIn.map((t) => t.pval.aval);
		const avalsOut = abstractEvalRules[primitive](avalsIn, params);
		const recipe = {
			type: "JaxprEqn",
			prim: primitive,
			tracersIn,
			params,
			avalsOut,
			tracerRefsOut: []
		};
		const tracersOut = avalsOut.map((aval, i) => {
			if (i > 0) tracersIn.forEach((t) => t.ref);
			return new PartialEvalTracer(this, PartialVal.unknown(aval), recipe);
		});
		recipe.tracerRefsOut = tracersOut.map((t) => new WeakRef(t));
		return tracersOut;
	}
	/**
	* Evaluate a Jaxpr on a set of PartialEvalTracers, computing as many known
	* values as possible (with JIT) and forwarding the unknown ones.
	*
	* Used when encountering a Jit rule during the trace.
	*/
	#partialEvalJaxpr(name, jaxpr, numConsts, tracers) {
		jaxpr = jaxpr.flatten();
		const inUnknowns = tracers.map((t) => !t.pval.isKnown);
		const { jaxpr1, jaxpr2, outUnknowns, numRes } = partialEvalJaxpr(jaxpr, inUnknowns);
		const [knownTracers, unknownTracers] = require_backend.partitionList(inUnknowns, tracers);
		const outs1Res = bind(Primitive.Jit, knownTracers.map((t) => t.ref.fullLower()), {
			name: `${name}_peval`,
			jaxpr: jaxpr1,
			numConsts: 0
		});
		const outs1 = outs1Res.slice(0, jaxpr1.outs.length - numRes);
		const res = outs1Res.slice(jaxpr1.outs.length - numRes);
		const resTracers = res.map((x) => this.instantiateConst(fullRaise(this, x)));
		const recipe = {
			type: "JaxprEqn",
			prim: Primitive.Jit,
			tracersIn: resTracers.concat(unknownTracers),
			params: {
				name: `${name}_resid`,
				jaxpr: jaxpr2,
				numConsts: 0
			},
			avalsOut: jaxpr2.outs.map((x) => x.aval),
			tracerRefsOut: []
		};
		const outs2 = jaxpr2.outs.map((x, i$1) => {
			if (i$1 > 0) recipe.tracersIn.forEach((t) => t.ref);
			return new PartialEvalTracer(this, PartialVal.unknown(x.aval), recipe);
		});
		recipe.tracerRefsOut = outs2.map((t) => new WeakRef(t));
		let i = 0;
		let j = 0;
		return outUnknowns.map((unk) => unk ? outs2[j++] : outs1[i++]);
	}
	/**
	* Partial eval for Scan primitive.
	*
	* When scan is encountered during partial evaluation (e.g., inside JVP for VJP):
	* - If all inputs are known, just run the scan
	* - If this is a JVP'd scan (doubled carry/xs), we can split primal (known)
	*   from tangent (unknown) outputs
	* - Otherwise, mark all outputs as unknown
	*/
	#partialEvalScan(params, tracers) {
		const { numConsts: _numConsts, numCarry } = params;
		const isKnown = tracers.map((t) => t.pval.isKnown);
		const hasUnknown = isKnown.some((k) => !k);
		if (!hasUnknown) {
			const inputs = tracers.map((t) => t.fullLower());
			return bind(Primitive.Scan, inputs, params);
		}
		const avalsIn = tracers.map((t) => t.pval.aval);
		const avalsOut = abstractEvalRules[Primitive.Scan](avalsIn, params);
		const numY = avalsOut.length - numCarry;
		const isJvpScan = numCarry % 2 === 0 && numY % 2 === 0;
		if (!isJvpScan) {
			const tracersIn$1 = tracers.map((t) => this.instantiateConst(t));
			const recipe$1 = {
				type: "JaxprEqn",
				prim: Primitive.Scan,
				tracersIn: tracersIn$1,
				params,
				avalsOut,
				tracerRefsOut: []
			};
			const tracersOut$1 = avalsOut.map((aval, i) => {
				if (i > 0) tracersIn$1.forEach((t) => t.ref);
				return new PartialEvalTracer(this, PartialVal.unknown(aval), recipe$1);
			});
			recipe$1.tracerRefsOut = tracersOut$1.map((t) => new WeakRef(t));
			return tracersOut$1;
		}
		const numPrimalCarry = numCarry / 2;
		const numPrimalY = numY / 2;
		const synthesizedZeroInputs = [];
		const fullInputs = tracers.map((t) => {
			if (t.pval.isKnown) return t.pval.val.ref;
			else {
				const z = zeros(t.pval.aval.shape, { dtype: t.pval.aval.dtype });
				synthesizedZeroInputs.push(z);
				return z;
			}
		});
		const fullOuts = bind(Primitive.Scan, fullInputs, params);
		const tracersIn = tracers.map((t) => this.instantiateConst(t));
		const recipe = {
			type: "JaxprEqn",
			prim: Primitive.Scan,
			tracersIn,
			params,
			avalsOut,
			tracerRefsOut: []
		};
		const tracersOut = [];
		for (let i = 0; i < numPrimalCarry; i++) tracersOut.push(new PartialEvalTracer(this, PartialVal.known(fullOuts[i]), null));
		let isFirstUnknown = true;
		for (let i = numPrimalCarry; i < numCarry; i++) {
			fullOuts[i].dispose();
			if (!isFirstUnknown) tracersIn.forEach((t) => t.ref);
			isFirstUnknown = false;
			tracersOut.push(new PartialEvalTracer(this, PartialVal.unknown(avalsOut[i]), recipe));
		}
		for (let i = 0; i < numPrimalY; i++) tracersOut.push(new PartialEvalTracer(this, PartialVal.known(fullOuts[numCarry + i]), null));
		for (let i = numPrimalY; i < numY; i++) {
			fullOuts[numCarry + i].dispose();
			tracersIn.forEach((t) => t.ref);
			tracersOut.push(new PartialEvalTracer(this, PartialVal.unknown(avalsOut[numCarry + i]), recipe));
		}
		const retainedKnownOutputs = /* @__PURE__ */ new Set();
		for (let i = 0; i < numPrimalCarry; i++) retainedKnownOutputs.add(fullOuts[i]);
		for (let i = 0; i < numPrimalY; i++) retainedKnownOutputs.add(fullOuts[numCarry + i]);
		for (const inp of fullInputs) if (!retainedKnownOutputs.has(inp) && inp.refCount > 0) inp.dispose();
		recipe.tracerRefsOut = tracersOut.map((t) => t.pval.isKnown ? null : new WeakRef(t));
		return tracersOut;
	}
};
/** Partially evaluate a Jaxpr, returning an immediate and residual Jaxpr. */
function partialEvalJaxpr(jaxpr, inUnknowns, instantiate) {
	jaxpr = jaxpr.flatten();
	const knownIns = jaxpr.inBinders.filter((_, i) => !inUnknowns[i]);
	const knownVars = new Set(knownIns);
	const residuals = /* @__PURE__ */ new Set();
	const eqns1 = [];
	const eqns2 = [];
	for (const eqn of jaxpr.eqns) {
		if (eqn.primitive === Primitive.Jit) throw new TypeError("partialEvalJaxpr requires flattened Jaxpr");
		const hasUnknowns = eqn.inputs.some((x) => x instanceof Var && !knownVars.has(x));
		if (hasUnknowns) {
			for (const x of eqn.inputs) if (x instanceof Var && knownVars.has(x)) residuals.add(x);
			eqns2.push(eqn);
		} else {
			eqns1.push(eqn);
			for (const v of eqn.outBinders) knownVars.add(v);
		}
	}
	const outUnknowns = jaxpr.outs.map((x) => x instanceof Var && !knownVars.has(x));
	if (instantiate !== void 0) for (let i = 0; i < jaxpr.outs.length; i++) {
		const x = jaxpr.outs[i];
		if (instantiate[i] && !outUnknowns[i] && x instanceof Var) {
			residuals.add(x);
			outUnknowns[i] = true;
		}
	}
	const residualsL = Array.from(residuals);
	const [ins1, ins2] = require_backend.partitionList(inUnknowns, jaxpr.inBinders);
	const [outs1, outs2] = require_backend.partitionList(outUnknowns, jaxpr.outs);
	const jaxpr1 = new Jaxpr(ins1, eqns1, outs1.concat(residualsL));
	const jaxpr2 = new Jaxpr(residualsL.concat(ins2), eqns2, outs2);
	return {
		jaxpr1,
		jaxpr2,
		outUnknowns,
		numRes: residualsL.length
	};
}
/**
* Convert the graph representation of a partial eval to a standard Jaxpr.
* Also called `tracers_to_jaxpr()` in JAX.
*/
function partialEvalGraphToJaxpr(tracersIn, tracersOut) {
	const tracerToVar = /* @__PURE__ */ new Map();
	const constToVar = /* @__PURE__ */ new Map();
	const processedEqns = /* @__PURE__ */ new Set();
	const constPETracers = [];
	const eqns = [];
	for (const t of tracersIn) tracerToVar.set(t, new Var(ShapedArray.fromAval(t.aval)));
	for (const t of require_backend.toposort(tracersOut, (t$1) => t$1.recipe?.type === "JaxprEqn" ? t$1.recipe.tracersIn : [])) {
		if (!t.recipe) throw new TypeError("Tracer is missing a recipe, cannot construct Jaxpr");
		if (t.recipe.type === "LambdaBinding") {
			if (!tracersIn.includes(t)) throw new TypeError("LambdaBinding tracer not in input list");
		} else if (t.recipe.type === "Const") {
			const val = t.recipe.val;
			let binder = constToVar.get(val);
			if (!binder) {
				binder = new Var(ShapedArray.fromAval(val.aval));
				constToVar.set(val, binder);
			}
			tracerToVar.set(t, binder);
			constPETracers.push(t);
		} else if (t.recipe.type === "JaxprEqn") {
			if (!processedEqns.has(t.recipe)) {
				processedEqns.add(t.recipe);
				const tracersIn$1 = t.recipe.tracersIn.map((t$1) => tracerToVar.get(t$1));
				const outBinders = t.recipe.avalsOut.map((aval) => new Var(aval));
				for (let i = 0; i < outBinders.length; i++) {
					const ref$1 = t.recipe.tracerRefsOut[i];
					const tracerOut = ref$1?.deref?.();
					if (tracerOut) tracerToVar.set(tracerOut, outBinders[i]);
				}
				eqns.push(new JaxprEqn(t.recipe.prim, tracersIn$1, t.recipe.params, outBinders));
			}
		}
	}
	const [consts, constvars] = require_backend.unzip2(constToVar.entries());
	const inBinders = [...constvars, ...tracersIn.map((t) => tracerToVar.get(t))];
	const outVars = tracersOut.map((t) => tracerToVar.get(t));
	let jaxpr = new Jaxpr(inBinders, eqns, outVars);
	typecheckJaxpr(jaxpr);
	for (const c of consts) c.ref;
	for (const t of constPETracers) t.dispose();
	for (const t of tracersIn) if (!t.pval.isKnown) t.dispose();
	for (const t of tracersOut) if (!t.pval.isKnown) t.dispose();
	jaxpr = jaxpr.simplify();
	if (require_backend.DEBUG >= 5) console.info("jaxpr from partial evaluation:\n" + jaxpr.toString());
	return new ClosedJaxpr(jaxpr, consts);
}
/** Marker type for pullback, used by transpose rules. */
var UndefPrimal = class {
	aval;
	constructor(aval) {
		this.aval = ShapedArray.fromAval(aval);
	}
};
/**
* Helper to get or compute a primal (known) variable's value during transpose.
* For intermediate variables that are known (computed from only known inputs),
* we need to evaluate the equations that produce them.
*/
function getOrComputePrimal(jaxpr, knownVars, knownPrimals, v, internalArrays) {
	if (knownPrimals.has(v)) {
		const r$1 = knownPrimals.get(v).ref;
		if (internalArrays) internalArrays.add(r$1);
		return r$1;
	}
	const eqn = jaxpr.eqns.find((eq) => eq.outBinders.some((out) => out === v));
	if (!eqn) throw new Error(`Internal error: could not find equation producing variable`);
	const inputVals = eqn.inputs.map((inp) => inp instanceof Lit ? array(inp.value, { dtype: inp.dtype }) : getOrComputePrimal(jaxpr, knownVars, knownPrimals, inp, internalArrays));
	const results = bind(eqn.primitive, inputVals, eqn.params);
	for (let i = 0; i < eqn.outBinders.length; i++) knownPrimals.set(eqn.outBinders[i], results[i]);
	const result = knownPrimals.get(v);
	if (!result) throw new Error(`Internal error: variable not produced by equation`);
	const r = result.ref;
	if (internalArrays) internalArrays.add(r);
	return r;
}
/**
* Evaluate the backward pass over a linearized Jaxpr (pullback of cotangents).
*
* Will raise a TypeError if the provided Jaxpr is not a linear function of its,
* inputs, as general expressions cannot be transposed.
*/
function evalJaxprTransposed(jaxpr, args, cotangents) {
	const knownVars = /* @__PURE__ */ new Set();
	for (let i = 0; i < jaxpr.inBinders.length; i++) if (!(args[i] instanceof UndefPrimal)) knownVars.add(jaxpr.inBinders[i]);
	for (const eqn of jaxpr.eqns) {
		const allInputsKnown = eqn.inputs.every((v) => v instanceof Lit || knownVars.has(v));
		if (allInputsKnown) for (const outVar of eqn.outBinders) knownVars.add(outVar);
	}
	const knownPrimals = /* @__PURE__ */ new Map();
	const argPrimals = /* @__PURE__ */ new Set();
	const argPrimalInitRc = /* @__PURE__ */ new Map();
	for (let i = 0; i < jaxpr.inBinders.length; i++) if (!(args[i] instanceof UndefPrimal)) {
		const arg = args[i];
		knownPrimals.set(jaxpr.inBinders[i], arg);
		argPrimals.add(jaxpr.inBinders[i]);
		const concrete = unwrapToConcreteArray(arg);
		argPrimalInitRc.set(jaxpr.inBinders[i], concrete instanceof Array$1 ? concrete.refCount : arg.refCount);
	}
	const ctStore = /* @__PURE__ */ new Map();
	const internalArrays = /* @__PURE__ */ new Set();
	const externalCts = /* @__PURE__ */ new Set();
	for (const ct of cotangents) if (ct instanceof Tracer) externalCts.add(ct);
	const readCotangent = (v) => {
		const ct = ctStore.get(v);
		if (ct) {
			ctStore.delete(v);
			return ct;
		} else {
			const z = zeros(v.aval.shape, { dtype: v.aval.dtype });
			internalArrays.add(z);
			return z;
		}
	};
	const writeCotangent = (v, ct) => {
		if (ct !== null) {
			if (!externalCts.has(ct)) internalArrays.add(ct);
			const oldCt = ctStore.get(v);
			if (oldCt) {
				const sum$1 = add$1(oldCt, ct);
				internalArrays.add(sum$1);
				ctStore.set(v, sum$1);
			} else ctStore.set(v, ct);
		}
	};
	for (let i = 0; i < jaxpr.outs.length; i++) {
		const v = jaxpr.outs[i];
		if (v instanceof Var) writeCotangent(v, cotangents[i]);
	}
	for (let i = jaxpr.eqns.length - 1; i >= 0; i--) {
		const eqn = jaxpr.eqns[i];
		const allInputsKnown = eqn.inputs.every((v) => v instanceof Lit || knownVars.has(v));
		if (allInputsKnown) continue;
		const primalsIn = eqn.inputs.map((v) => {
			if (v instanceof Lit) {
				const lit = array(v.value, { dtype: v.dtype });
				internalArrays.add(lit);
				return lit;
			}
			return knownVars.has(v) ? getOrComputePrimal(jaxpr, knownVars, knownPrimals, v, internalArrays) : new UndefPrimal(v.aval);
		});
		const cotangentsOut = eqn.outBinders.map(readCotangent);
		const rule = transposeRules[eqn.primitive];
		if (!rule) throw new TypeError(`Backward pass not implemented for ${eqn.primitive}`);
		const cotangentsIn = rule(cotangentsOut, primalsIn, eqn.params);
		for (let j = 0; j < eqn.inputs.length; j++) {
			const v = eqn.inputs[j];
			if (v instanceof Var && !knownVars.has(v)) writeCotangent(v, cotangentsIn[j]);
			else if (cotangentsIn[j] !== null) throw new Error("internal: cotangent should be null");
		}
	}
	const results = [];
	for (let i = 0; i < jaxpr.inBinders.length; i++) if (args[i] instanceof UndefPrimal) results.push(readCotangent(jaxpr.inBinders[i]));
	if (!insideAbstractTrace()) {
		for (const r of results) if (r instanceof Array$1) r._flushPendingSync();
	}
	for (const v of argPrimals) {
		const val = knownPrimals.get(v);
		const initialRc = argPrimalInitRc.get(v);
		if (val && initialRc !== void 0) {
			const concrete = unwrapToConcreteArray(val);
			const currentRc = concrete instanceof Array$1 ? concrete.refCount : val.refCount;
			const excess = currentRc - initialRc;
			for (let i = 0; i < excess; i++) try {
				val.dispose();
			} catch {
				break;
			}
		}
	}
	if (!insideAbstractTrace()) {
		const returnedSet = new Set(results);
		for (const [v, t] of knownPrimals.entries()) if (!argPrimals.has(v) && !returnedSet.has(t)) try {
			while (t.refCount > 0) t.dispose();
		} catch {}
		for (const arr of internalArrays) if (!returnedSet.has(arr) && !externalCts.has(arr)) {
			let isArgPrimal = false;
			for (const v of argPrimals) if (knownPrimals.get(v) === arr) {
				isArgPrimal = true;
				break;
			}
			if (!isArgPrimal) try {
				arr.dispose();
			} catch {}
		}
	}
	return results;
}
/**
* Inverse operation of `generalBroadcast()` for backpropagation.
*
* `x` has the shape of the result of an operation that was broadcasted with
* `target` (it's a cotangent during backprop). Returns a tracer with rank and
* shape equal to `target`.
*/
function unbroadcast(x, target) {
	const shape$1 = target.aval.shape;
	const extraDims = x.ndim > shape$1.length ? require_backend.range(x.ndim - shape$1.length) : [];
	if (x.ndim < shape$1.length) throw new Error(`unbroadcast: x.ndim (${x.shape}) < target.ndim (${shape$1})`);
	const unsqueeze = [];
	const keptReduceDims = [];
	for (let i = 0; i < shape$1.length; i++) {
		const indexFromEnd = shape$1.length - i;
		const indexInX = x.ndim - indexFromEnd;
		const xLen = x.shape[indexInX];
		if (xLen > 1 && shape$1[i] === 1) {
			unsqueeze.push(i);
			keptReduceDims.push(indexInX);
		} else if (shape$1[i] !== xLen) throw new Error("internal: unbroadcast shape mismatch");
	}
	const reductionDims = [...extraDims, ...keptReduceDims];
	if (reductionDims.length === 0) return x;
	let result = x.sum(reductionDims);
	if (!require_backend.deepEqual(result.shape, shape$1)) {
		const sumResult = result;
		result = broadcast(sumResult, shape$1, unsqueeze);
		sumResult.dispose();
	}
	return result;
}
var NonlinearError = class extends TypeError {
	constructor(primitive) {
		super(`Nonlinear operation in backward pass for ${primitive}`);
	}
};
const transposeRules = {
	[Primitive.Mul]([ct], [x, y]) {
		if (x instanceof UndefPrimal === y instanceof UndefPrimal) throw new NonlinearError(Primitive.Mul);
		if (x instanceof UndefPrimal) {
			const prod$2 = mul(ct, y);
			const result = unbroadcast(prod$2, x);
			if (result !== prod$2) prod$2.dispose();
			return [result, null];
		} else {
			const prod$2 = mul(x, ct);
			const result = unbroadcast(prod$2, y);
			if (result !== prod$2) prod$2.dispose();
			return [null, result];
		}
	},
	[Primitive.Neg]([ct], [x]) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Neg);
		return [neg(ct)];
	},
	[Primitive.Add]([ct], [x, y]) {
		if (!(x instanceof UndefPrimal || y instanceof UndefPrimal)) throw new NonlinearError(Primitive.Add);
		if (x instanceof UndefPrimal && y instanceof UndefPrimal) return [unbroadcast(ct, x), unbroadcast(ct, y)];
		return x instanceof UndefPrimal ? (y.dispose(), [unbroadcast(ct, x), null]) : (x.dispose(), [null, unbroadcast(ct, y)]);
	},
	[Primitive.Reduce]([ct], [x], { op, axis }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Reduce);
		if (op === require_backend.AluOp.Add) return [broadcast(ct, x.aval.shape, axis)];
		else throw new NonlinearError(Primitive.Reduce);
	},
	[Primitive.Pool]([ct], [x], { window, strides }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Pool);
		return bind(Primitive.PoolTranspose, [ct], {
			inShape: x.aval.shape,
			window,
			strides
		});
	},
	[Primitive.PoolTranspose]([ct], [x], { window, strides }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.PoolTranspose);
		return bind(Primitive.Pool, [ct], {
			window,
			strides
		});
	},
	[Primitive.Dot]([ct], [x, y]) {
		if (x instanceof UndefPrimal === y instanceof UndefPrimal) throw new NonlinearError(Primitive.Dot);
		const axisSize = require_backend.generalBroadcast(x.aval.shape, y.aval.shape).slice(-1)[0];
		const ctBroad = broadcast(ct, ct.shape.concat(axisSize), [-1]);
		if (x instanceof UndefPrimal) {
			const prod$2 = mul(ctBroad, y);
			const result = unbroadcast(prod$2, x);
			if (result !== prod$2) prod$2.dispose();
			ctBroad.dispose();
			return [result, null];
		} else {
			const prod$2 = mul(x, ctBroad);
			const result = unbroadcast(prod$2, y);
			if (result !== prod$2) prod$2.dispose();
			ctBroad.dispose();
			return [null, result];
		}
	},
	[Primitive.Conv]([ct], [lhs, rhs], params) {
		if (lhs instanceof UndefPrimal === rhs instanceof UndefPrimal) throw new NonlinearError(Primitive.Conv);
		const v = params.vmapDims;
		const rev01 = [
			...require_backend.range(v),
			v + 1,
			v,
			...require_backend.range(v + 2, ct.ndim)
		];
		if (lhs instanceof UndefPrimal) try {
			var _usingCtx3 = (0, import_usingCtx$7.default)();
			const kernelTransposed = _usingCtx3.u(transpose$1(rhs, rev01));
			const kernel = _usingCtx3.u(flip$1(kernelTransposed, require_backend.range(v + 2, kernelTransposed.ndim)));
			const result = conv$1(ct, kernel, {
				vmapDims: v,
				strides: params.lhsDilation,
				padding: params.padding.map(([pl, _pr], i) => {
					const dilatedKernel = (kernel.shape[i + v + 2] - 1) * params.rhsDilation[i] + 1;
					const dilatedCt = (ct.shape[i + v + 2] - 1) * params.strides[i] + 1;
					const padBefore = dilatedKernel - 1 - pl;
					const dilatedLhs = (lhs.aval.shape[i + v + 2] - 1) * params.lhsDilation[i] + 1;
					const padAfter = dilatedLhs + dilatedKernel - 1 - dilatedCt - padBefore;
					return [padBefore, padAfter];
				}),
				lhsDilation: params.strides,
				rhsDilation: params.rhsDilation
			});
			return [result, null];
		} catch (_) {
			_usingCtx3.e = _;
		} finally {
			_usingCtx3.d();
		}
		else try {
			var _usingCtx4 = (0, import_usingCtx$7.default)();
			const newLhs = _usingCtx4.u(transpose$1(lhs, rev01));
			const newRhs = _usingCtx4.u(transpose$1(ct, rev01));
			const convResult = _usingCtx4.u(conv$1(newLhs, newRhs, {
				vmapDims: v,
				strides: params.rhsDilation,
				padding: params.padding.map(([pl, _pr], i) => {
					const dilatedLhs = (lhs.aval.shape[i + v + 2] - 1) * params.lhsDilation[i] + 1;
					const dilatedKernel = (rhs.aval.shape[i + v + 2] - 1) * params.rhsDilation[i] + 1;
					const dilatedCt = (ct.shape[i + v + 2] - 1) * params.strides[i] + 1;
					const padFromLhs = dilatedCt - dilatedLhs;
					const padFromRhs = dilatedKernel - pl - 1;
					return [pl, padFromLhs + padFromRhs];
				}),
				lhsDilation: params.lhsDilation,
				rhsDilation: params.strides
			}));
			const result = transpose$1(convResult, rev01);
			return [null, result];
		} catch (_) {
			_usingCtx4.e = _;
		} finally {
			_usingCtx4.d();
		}
	},
	[Primitive.Where]([ct], [cond, x, y]) {
		const cts = [
			null,
			null,
			null
		];
		if (cond instanceof UndefPrimal) throw new NonlinearError(Primitive.Where);
		if (x instanceof UndefPrimal) {
			const masked = where$1(cond, ct, 0);
			cts[1] = unbroadcast(masked, x);
			if (cts[1] !== masked) masked.dispose();
		} else x.dispose();
		if (y instanceof UndefPrimal) {
			const masked = where$1(cond, 0, ct);
			cts[2] = unbroadcast(masked, y);
			if (cts[2] !== masked) masked.dispose();
		} else y.dispose();
		return cts;
	},
	[Primitive.Concatenate]([ct], inputs, { axis }) {
		if (inputs.some((x) => !(x instanceof UndefPrimal))) throw new NonlinearError(Primitive.Concatenate);
		const sizes = inputs.map((x) => x.aval.shape[axis]);
		return split$2(ct, axis, sizes);
	},
	[Primitive.Split](cts, [x], { axis }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Split);
		return [concatenate$1(cts, axis)];
	},
	[Primitive.Gather]([ct], [x, ...indices], { axis, outDim }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Gather);
		if (indices.some((i) => i instanceof UndefPrimal)) throw new NonlinearError(Primitive.Gather);
		if (indices.length === 1 && axis.length === 1 && ct.shape[axis[0]] === x.aval.shape[axis[0]]) {
			const idx = indices[0];
			const [sortedVals, invIdx] = argsort$1(idx);
			const result = gather(ct, [invIdx], axis, outDim);
			sortedVals.dispose();
			invIdx.dispose();
			return [result, null];
		}
		throw new Error("Gather transpose rule is only implemented for permutation gathers. General case (duplicate indices) requires a scatter_add primitive.");
	},
	[Primitive.Transpose]([ct], [x], { perm }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Transpose);
		return [transpose$1(ct, require_backend.invertPermutation(perm))];
	},
	[Primitive.Broadcast]([ct], [x], { axis }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Broadcast);
		return [reduce(ct, require_backend.AluOp.Add, axis)];
	},
	[Primitive.Reshape]([ct], [x], _) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Reshape);
		return [reshape$1(ct, x.aval.shape)];
	},
	[Primitive.Flip]([ct], [x], { axis }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Flip);
		return [flip$1(ct, axis)];
	},
	[Primitive.Shrink]([ct], [x], { slice }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Shrink);
		const width = slice.map(([s, e$1], i) => [s, x.aval.shape[i] - e$1]);
		return [pad$1(ct, width)];
	},
	[Primitive.Pad]([ct], [x], { width }) {
		if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Pad);
		const slice = width.map(([s, _e], i) => [s, s + x.aval.shape[i]]);
		return [shrink(ct, slice)];
	},
	[Primitive.TriangularSolve]([ct], [a, b], { unitDiagonal }) {
		try {
			var _usingCtx5 = (0, import_usingCtx$7.default)();
			if (a instanceof UndefPrimal || !(b instanceof UndefPrimal)) throw new NonlinearError(Primitive.TriangularSolve);
			const aT = _usingCtx5.u(moveaxis(a, -2, -1));
			const ctB = triangularSolve$1(aT, ct, {
				lower: true,
				unitDiagonal
			});
			return [null, ctB];
		} catch (_) {
			_usingCtx5.e = _;
		} finally {
			_usingCtx5.d();
		}
	},
	[Primitive.Jit](cts, args, { name, jaxpr }) {
		const undefPrimals = args.map((x) => x instanceof UndefPrimal);
		const newJaxpr = transposeJaxpr(jaxpr, undefPrimals);
		const residuals = args.filter((x, i$1) => !undefPrimals[i$1]);
		const outs = bind(Primitive.Jit, [
			...newJaxpr.consts.map((c) => c.ref),
			...residuals,
			...cts
		], {
			name: `${name}_t`,
			jaxpr: newJaxpr.jaxpr,
			numConsts: newJaxpr.consts.length
		});
		let i = 0;
		return undefPrimals.map((isUndef) => isUndef ? outs[i++] : null);
	},
	[Primitive.DynamicUpdateSlice]() {
		throw new Error("DynamicUpdateSlice transpose: not yet implemented");
	},
	[Primitive.Scan](cts, args, { jaxpr, numCarry, numConsts, length, reverse, checkpoint }) {
		const numX = args.length - numConsts - numCarry;
		const numY = cts.length - numCarry;
		const isJvpScan = numCarry % 2 === 0 && numY % 2 === 0 && numX % 2 === 0;
		const numPrimalCarry = isJvpScan ? numCarry / 2 : 0;
		const numPrimalX = isJvpScan ? numX / 2 : 0;
		const numPrimalY = isJvpScan ? numY / 2 : 0;
		const undefMask = args.map((x, i) => {
			if (x instanceof UndefPrimal) return true;
			if (!isJvpScan) return false;
			if (i < numConsts) return false;
			else if (i < numConsts + numCarry) {
				const carryIdx = i - numConsts;
				return carryIdx >= numPrimalCarry;
			} else {
				const xIdx = i - numConsts - numCarry;
				return xIdx >= numPrimalX;
			}
		});
		const bodyNumConsts = numConsts;
		const bodyNumCarry = numCarry;
		const bodyUndefPrimals = [];
		for (let i = 0; i < jaxpr.inBinders.length; i++) if (i < bodyNumConsts) bodyUndefPrimals.push(undefMask[i]);
		else if (i < bodyNumConsts + bodyNumCarry) bodyUndefPrimals.push(undefMask[numConsts + (i - bodyNumConsts)]);
		else bodyUndefPrimals.push(undefMask[numConsts + numCarry + (i - bodyNumConsts - bodyNumCarry)]);
		const constArgs = args.slice(0, numConsts);
		const carryArgs = args.slice(numConsts, numConsts + numCarry);
		const xsArgs = args.slice(numConsts + numCarry);
		const constResiduals = constArgs.filter((_, i) => !undefMask[i]);
		const carryResiduals = carryArgs.filter((_, i) => !undefMask[numConsts + i]);
		const xsResiduals = xsArgs.filter((_, i) => !undefMask[numConsts + numCarry + i]);
		const actualNumPrimalCarry = isJvpScan ? numPrimalCarry : carryArgs.map((_, i) => !undefMask[numConsts + i]).filter((x) => x).length;
		if (actualNumPrimalCarry === 0 || carryResiduals.length === 0) throw new Error("Scan transpose: no carry residuals available. grad() through scan requires primal carry values to be available as residuals.");
		const forwardInTypes = jaxpr.inBinders.filter((_, i) => !bodyUndefPrimals[i]).map((v) => v.aval);
		const { jaxpr: primalForwardJaxpr } = makeJaxpr$1((...primalInputs) => {
			const fullInputs = [];
			let primalIdx = 0;
			for (let i = 0; i < jaxpr.inBinders.length; i++) if (bodyUndefPrimals[i]) {
				const aval = jaxpr.inBinders[i].aval;
				fullInputs.push(zeros(aval.shape, { dtype: aval.dtype }));
			} else fullInputs.push(primalInputs[primalIdx++].ref);
			const outs = evalJaxpr(jaxpr, fullInputs);
			const primalCarryOuts = outs.slice(0, numPrimalCarry);
			const primalYOuts = outs.slice(numCarry, numCarry + Math.floor(numY / 2));
			for (let i = numPrimalCarry; i < numCarry; i++) outs[i].dispose();
			for (let i = numCarry + Math.floor(numY / 2); i < outs.length; i++) outs[i].dispose();
			return [...primalCarryOuts, ...primalYOuts];
		}, { validateRefs: false })(...forwardInTypes);
		const runOneForwardStep = (iter, carry) => {
			const dataIdx = reverse ? length - 1 - iter : iter;
			const xSlices = [];
			for (const xs of xsResiduals) {
				const slice = shrink(xs, [[dataIdx, dataIdx + 1], ...xs.shape.slice(1).map((_, i) => [0, xs.shape[i + 1]])]);
				const reshaped = reshape$1(slice, xs.shape.slice(1));
				slice.dispose();
				xSlices.push(reshaped);
			}
			const forwardInputs = [
				...constResiduals.map((c) => c.ref),
				...carry.map((c) => c.ref),
				...xSlices
			];
			const forwardOuts = evalJaxpr(primalForwardJaxpr.jaxpr, [...primalForwardJaxpr.consts.map((c) => c.ref), ...forwardInputs]);
			const newCarry = forwardOuts.slice(0, numPrimalCarry);
			for (let i = numPrimalCarry; i < forwardOuts.length; i++) forwardOuts[i].dispose();
			for (const x of xSlices) if (x.refCount > 0) x.dispose();
			return newCarry;
		};
		const useCheckpointing = checkpoint !== false;
		const segmentSize = useCheckpointing ? typeof checkpoint === "number" ? checkpoint : Math.max(1, Math.ceil(Math.sqrt(length))) : length;
		const allCarries = useCheckpointing ? null : [];
		const checkpointCarries = useCheckpointing ? /* @__PURE__ */ new Map() : null;
		{
			let currentCarry = carryResiduals.map((c) => c.ref);
			if (allCarries) allCarries.push(currentCarry.map((c) => c.ref));
			else checkpointCarries.set(0, currentCarry.map((c) => c.ref));
			for (let iter = 0; iter < length; iter++) {
				const newCarry = runOneForwardStep(iter, currentCarry);
				for (const c of currentCarry) c.dispose();
				currentCarry = newCarry;
				if (allCarries) allCarries.push(currentCarry.map((c) => c.ref));
				else if ((iter + 1) % segmentSize === 0) checkpointCarries.set(iter + 1, currentCarry.map((c) => c.ref));
			}
			for (const c of currentCarry) c.dispose();
		}
		const numTangentConsts = numConsts - constResiduals.length;
		const numTangentCarry = numCarry - numPrimalCarry;
		const numTangentX = numX - numPrimalX;
		const tangentBodyInAvals = [...jaxpr.inBinders.filter((_, i) => !bodyUndefPrimals[i]).map((v) => v.aval), ...jaxpr.inBinders.filter((_, i) => bodyUndefPrimals[i]).map((v) => v.aval)];
		const { jaxpr: tangentBody } = makeJaxpr$1((...tangentBodyArgs) => {
			const numPrimalInputs = jaxpr.inBinders.filter((_, i) => !bodyUndefPrimals[i]).length;
			const primalResiduals = tangentBodyArgs.slice(0, numPrimalInputs);
			const tangentInputs = tangentBodyArgs.slice(numPrimalInputs);
			const fullInputs = [];
			let primalIdx = 0;
			let tangentIdx = 0;
			for (let i = 0; i < jaxpr.inBinders.length; i++) if (bodyUndefPrimals[i]) fullInputs.push(tangentInputs[tangentIdx++].ref);
			else fullInputs.push(primalResiduals[primalIdx++].ref);
			const fullOuts = evalJaxpr(jaxpr, fullInputs);
			const tangentOuts = [];
			for (let i = numPrimalCarry; i < numCarry; i++) tangentOuts.push(fullOuts[i]);
			for (let i = numCarry + numPrimalY; i < fullOuts.length; i++) tangentOuts.push(fullOuts[i]);
			for (let i = 0; i < numPrimalCarry; i++) fullOuts[i].dispose();
			for (let i = numCarry; i < numCarry + numPrimalY; i++) fullOuts[i].dispose();
			return tangentOuts;
		}, { validateRefs: false })(...tangentBodyInAvals);
		const tangentBodyUndefPrimals = [...Array(tangentBody.jaxpr.inBinders.length - (numTangentConsts + numTangentCarry + numTangentX)).fill(false), ...Array(numTangentConsts + numTangentCarry + numTangentX).fill(true)];
		const transposedBody = transposeJaxpr(tangentBody.jaxpr, tangentBodyUndefPrimals);
		const ctCarryAll = cts.slice(0, numCarry);
		const ctYsAll = cts.slice(numCarry);
		let ctCarryRunning = ctCarryAll.slice(numPrimalCarry).map((c) => c.ref);
		for (let i = 0; i < numPrimalCarry; i++) ctCarryAll[i].dispose();
		const ctXsAccum = [];
		for (let i = 0; i < numTangentX; i++) ctXsAccum.push([]);
		let ctConstsAccum = null;
		const runOneBackwardStep = (iter, primalCarry) => {
			const dataIdx = reverse ? length - 1 - iter : iter;
			const xSlices = [];
			for (const xs of xsResiduals) {
				const slice = shrink(xs, [[dataIdx, dataIdx + 1], ...xs.shape.slice(1).map((_, i) => [0, xs.shape[i + 1]])]);
				const reshaped = reshape$1(slice, xs.shape.slice(1));
				slice.dispose();
				xSlices.push(reshaped);
			}
			const ctYSlices = [];
			for (let i = Math.floor(numY / 2); i < ctYsAll.length; i++) {
				const ctY = ctYsAll[i];
				const slice = shrink(ctY, [[dataIdx, dataIdx + 1], ...ctY.shape.slice(1).map((_, j) => [0, ctY.shape[j + 1]])]);
				const reshaped = reshape$1(slice, ctY.shape.slice(1));
				slice.dispose();
				ctYSlices.push(reshaped);
			}
			const bodyOutCotangents = [];
			bodyOutCotangents.push(...ctCarryRunning.map((c) => c.ref));
			bodyOutCotangents.push(...ctYSlices);
			const transposedInputs = [
				...transposedBody.consts.map((c) => c.ref),
				...constResiduals.map((c) => c.ref),
				...primalCarry.map((c) => c.ref),
				...xSlices,
				...bodyOutCotangents
			];
			const transposedOuts = evalJaxpr(transposedBody.jaxpr, transposedInputs);
			let outIdx = 0;
			const ctConstsIter = [];
			for (let i = 0; i < numTangentConsts; i++) ctConstsIter.push(transposedOuts[outIdx++]);
			const ctCarryNew = [];
			const numTangentCarryLocal = numCarry - numPrimalCarry;
			for (let i = 0; i < numTangentCarryLocal; i++) ctCarryNew.push(transposedOuts[outIdx++]);
			const ctXIter = [];
			for (let i = 0; i < numTangentX; i++) ctXIter.push(transposedOuts[outIdx++]);
			if (ctConstsAccum === null) ctConstsAccum = ctConstsIter;
			else {
				const next = [];
				for (let i = 0; i < ctConstsAccum.length; i++) {
					const summed = add$1(ctConstsAccum[i], ctConstsIter[i]);
					ctConstsAccum[i].dispose();
					ctConstsIter[i].dispose();
					next.push(summed);
				}
				ctConstsAccum = next;
			}
			for (let i = 0; i < numTangentX; i++) ctXsAccum[i].push(ctXIter[i]);
			for (const c of ctCarryRunning) c.dispose();
			ctCarryRunning = ctCarryNew;
		};
		if (useCheckpointing) {
			const numSegments = Math.ceil(length / segmentSize);
			for (let seg = numSegments - 1; seg >= 0; seg--) {
				const segStart = seg * segmentSize;
				const segEnd = Math.min(segStart + segmentSize, length);
				const segCarries = [];
				let carry = checkpointCarries.get(segStart).map((c) => c.ref);
				segCarries.push(carry.map((c) => c.ref));
				for (let iter = segStart; iter < segEnd - 1; iter++) {
					const newCarry = runOneForwardStep(iter, carry);
					for (const c of carry) c.dispose();
					carry = newCarry;
					segCarries.push(carry.map((c) => c.ref));
				}
				for (const c of carry) c.dispose();
				for (let iter = segEnd - 1; iter >= segStart; iter--) {
					const localIdx = iter - segStart;
					runOneBackwardStep(iter, segCarries[localIdx]);
					for (const c of segCarries[localIdx]) c.dispose();
				}
				for (const c of checkpointCarries.get(segStart)) c.dispose();
				checkpointCarries.delete(segStart);
			}
			for (const [, carries] of checkpointCarries) for (const c of carries) c.dispose();
		} else {
			for (let iter = length - 1; iter >= 0; iter--) {
				runOneBackwardStep(iter, allCarries[iter]);
				for (const c of allCarries[iter]) c.dispose();
			}
			for (const c of allCarries[length]) c.dispose();
		}
		for (let i = Math.floor(numY / 2); i < ctYsAll.length; i++) ctYsAll[i].dispose();
		for (let i = 0; i < Math.floor(numY / 2); i++) ctYsAll[i].dispose();
		const ctXsStacked = [];
		for (let i = 0; i < numTangentX; i++) {
			const reversed = ctXsAccum[i].reverse();
			if (reverse) reversed.reverse();
			const expanded = reversed.map((ct) => broadcast(ct, [1, ...ct.shape], [0]));
			const stacked = concatenate$1(expanded, 0);
			const disposed = /* @__PURE__ */ new Set();
			for (const ct of expanded) if (!disposed.has(ct)) {
				disposed.add(ct);
				ct.dispose();
			}
			for (const ct of reversed) if (!disposed.has(ct)) {
				disposed.add(ct);
				ct.dispose();
			}
			ctXsStacked.push(stacked);
		}
		const actualUndefMask = args.map((x) => x instanceof UndefPrimal);
		const result = [];
		let ctConstIdx = 0;
		let ctCarryIdx = 0;
		let ctXIdx = 0;
		for (let i = 0; i < args.length; i++) {
			const isJvpTangent = undefMask[i];
			if (!actualUndefMask[i]) {
				if (isJvpTangent) if (i < numConsts) ctConstsAccum[ctConstIdx++].dispose();
				else if (i < numConsts + numCarry) ctCarryRunning[ctCarryIdx++].dispose();
				else ctXsStacked[ctXIdx++].dispose();
				result.push(null);
			} else if (i < numConsts) result.push(ctConstsAccum[ctConstIdx++]);
			else if (i < numConsts + numCarry) result.push(ctCarryRunning[ctCarryIdx++]);
			else result.push(ctXsStacked[ctXIdx++]);
		}
		if (ctConstsAccum) for (let i = ctConstIdx; i < ctConstsAccum.length; i++) ctConstsAccum[i].dispose();
		for (let i = ctCarryIdx; i < ctCarryRunning.length; i++) ctCarryRunning[i].dispose();
		for (let i = ctXIdx; i < ctXsStacked.length; i++) ctXsStacked[i].dispose();
		primalForwardJaxpr.dispose();
		transposedBody.dispose();
		for (const c of constResiduals) c.dispose();
		for (const c of carryResiduals) c.dispose();
		for (const c of xsResiduals) c.dispose();
		return result;
	}
};
const transposeJaxprCache = /* @__PURE__ */ new Map();
_registerJitCacheDisposer(() => {
	for (const inner$1 of transposeJaxprCache.values()) for (const cj of inner$1.values()) cj.dispose();
	transposeJaxprCache.clear();
});
function transposeJaxpr(jaxpr, undefPrimals) {
	const cacheKey = JSON.stringify(undefPrimals);
	const prevResult = transposeJaxprCache.get(jaxpr)?.get(cacheKey);
	if (prevResult) return prevResult;
	const { inTypes, outTypes } = typecheckJaxpr(jaxpr);
	const forwardInTypes = inTypes.filter((_, i) => !undefPrimals[i]);
	const { jaxpr: newJaxpr } = makeJaxpr$1((forwardIn, cotangents) => {
		const args = [];
		let forwardInIdx = 0;
		for (let i = 0; i < undefPrimals.length; i++) if (undefPrimals[i]) args.push(new UndefPrimal(inTypes[i]));
		else args.push(forwardIn[forwardInIdx++]);
		return evalJaxprTransposed(jaxpr, args, cotangents);
	}, { validateRefs: false })(forwardInTypes, outTypes);
	typecheckJaxpr(newJaxpr.jaxpr);
	if (!transposeJaxprCache.has(jaxpr)) transposeJaxprCache.set(jaxpr, /* @__PURE__ */ new Map());
	transposeJaxprCache.get(jaxpr).set(cacheKey, newJaxpr);
	return newJaxpr;
}
function vjpFlat(f, primalsIn, auxStore) {
	const { primalsOut, jaxpr, peIntermediates, literalIntermediates } = linearizeFlatUtil(f, primalsIn);
	const protectedVals = new Set(primalsOut);
	for (const c of jaxpr.consts) if (c.refCount <= 1) protectedVals.add(c);
	if (auxStore?.value != null) for (const arr of collectConcreteArrays(auxStore.value)) protectedVals.add(arr);
	disposePeIntermediates(peIntermediates, literalIntermediates, protectedVals);
	if (!insideAbstractTrace()) {
		for (const p of primalsOut) if (p instanceof Array$1) p._flushPendingSync();
	}
	const fVjp = (...cotangents) => {
		const transposeInputs = [...jaxpr.consts, ...primalsIn.map((t) => new UndefPrimal(t.aval))];
		return evalJaxprTransposed(jaxpr.jaxpr, transposeInputs, cotangents);
	};
	const dispose$1 = () => jaxpr.dispose();
	return [
		primalsOut,
		fVjp,
		dispose$1
	];
}
function vjp$1(f, primalsIn, { hasAux = false } = {}) {
	const [primalsInFlat, inTree] = flatten(primalsIn);
	let fFlat, outTree, aux;
	if (hasAux) [fFlat, outTree, aux] = flattenFunWithAux(f, inTree);
	else [fFlat, outTree] = flattenFun(f, inTree);
	const wrappedPrimals = primalsInFlat.map(pureArray);
	const [primalsOutFlat, fVjpFlat, innerDispose] = vjpFlat(fFlat, wrappedPrimals, hasAux ? aux : void 0);
	if (!insideAbstractTrace()) {
		const primalsOutSet = new Set(primalsOutFlat);
		for (let i = 0; i < wrappedPrimals.length; i++) if (wrappedPrimals[i] !== primalsInFlat[i] && !primalsOutSet.has(wrappedPrimals[i])) wrappedPrimals[i].dispose();
	}
	if (outTree.value === void 0) throw new Error("outTree was not set in vjp");
	const primalsOut = unflatten(outTree.value, primalsOutFlat);
	const fVjp = ((cotangentsOut) => {
		const [cotangentsOutFlat, outTree2] = flatten(cotangentsOut);
		if (!outTree.value.equals(outTree2)) throw new TreeMismatchError("vjp", outTree.value, outTree2);
		const wrappedCots = cotangentsOutFlat.map(pureArray);
		const cotangentsInFlat = fVjpFlat(...wrappedCots);
		if (!insideAbstractTrace()) {
			for (let i = 0; i < wrappedCots.length; i++) if (wrappedCots[i] !== cotangentsOutFlat[i]) wrappedCots[i].dispose();
		}
		return unflatten(inTree, cotangentsInFlat);
	});
	fVjp.dispose = innerDispose;
	fVjp[Symbol.dispose] = innerDispose;
	if (hasAux) return [
		primalsOut,
		fVjp,
		lowerAux(aux.value)
	];
	return [primalsOut, fVjp];
}
function grad$1(f, opts) {
	const valueAndGradFn = valueAndGrad$1(f, opts);
	return (...x) => {
		if (opts?.hasAux) {
			const [[y, aux], dx] = valueAndGradFn(...x);
			if (!insideAbstractTrace()) y.dispose();
			return [dx, aux];
		} else {
			const [y, dx] = valueAndGradFn(...x);
			if (!insideAbstractTrace()) y.dispose();
			return dx;
		}
	};
}
function valueAndGrad$1(f, opts) {
	const argnums = opts?.argnums ?? 0;
	const hasAux = opts?.hasAux ?? false;
	require_backend.checkInts(argnums);
	const argnumsSet = new Set(typeof argnums === "number" ? [argnums] : argnums);
	return (...x) => {
		if (x.length === 0) throw new Error("grad requires at least one argument to differentiate");
		const sgArrays = [];
		const sgOriginals = /* @__PURE__ */ new Set();
		for (let i = 0; i < x.length; i++) if (!argnumsSet.has(i)) x[i] = map((leaf) => {
			const sg = stopGradient(leaf);
			if (sg instanceof Tracer) {
				if (leaf instanceof Tracer) sgOriginals.add(leaf);
				sgArrays.push(sg);
			}
			return sg;
		}, x[i]);
		const [y, fVjp, aux] = vjp$1(f, x, { hasAux });
		if (!(y instanceof Tracer) || ndim$1(y) !== 0) {
			if (!insideAbstractTrace()) {
				fVjp.dispose();
				dispose(y);
				if (hasAux) dispose(aux);
				for (const a of sgArrays) if (!sgOriginals.has(a)) a.dispose();
			}
			throw new TypeError("grad requires a scalar output");
		}
		if (!require_backend.isFloatDtype(y.dtype)) {
			if (!insideAbstractTrace()) {
				fVjp.dispose();
				dispose(y);
				if (hasAux) dispose(aux);
				for (const a of sgArrays) if (!sgOriginals.has(a)) a.dispose();
			}
			throw new TypeError("grad only supports floating-point dtypes");
		}
		const seed = onesLike$1(y);
		const cts = fVjp(seed);
		let seedEscapes = false;
		for (const ct of cts) {
			if (ct === seed) {
				seedEscapes = true;
				break;
			}
			for (const arr of collectConcreteArrays(ct)) if (arr === seed) {
				seedEscapes = true;
				break;
			}
			if (seedEscapes) break;
		}
		const shouldDisposeSeed = !seedEscapes;
		if (!insideAbstractTrace()) {
			for (const a of sgArrays) if (!sgOriginals.has(a)) a.dispose();
		}
		fVjp.dispose();
		if (shouldDisposeSeed) {
			if (!insideAbstractTrace()) seed.dispose();
			else if (currentTraceLevel() === 1 && seed instanceof Array$1 && seed.refCount > 0) seed.dispose();
		}
		for (let i = 0; i < cts.length; i++) if (!argnumsSet.has(i)) dispose(cts[i]);
		const grads = typeof argnums === "number" ? cts[argnums] : argnums.map((i) => cts[i]);
		return hasAux ? [[y, aux], grads] : [y, grads];
	};
}
function jacrev$1(f) {
	return function jacobianReverse(x) {
		if (x.shape.length !== 1) throw new TypeError("jacrev only supports 1D inputs");
		const [size$1] = x.shape;
		const pullback = (ct) => {
			const [y, fVjp] = vjp$1(f, [x]);
			y.dispose();
			const [ret] = fVjp(ct);
			fVjp.dispose();
			return ret;
		};
		const eyeMatrix = eye(size$1, void 0, { dtype: x.dtype });
		const result = vmap$1(pullback, [1])(eyeMatrix);
		eyeMatrix.dispose();
		return result;
	};
}
function hessian$1(f) {
	return jacfwd$1(grad$1(f));
}

//#endregion
//#region src/library/numpy/einsum.ts
const bprod = (...xs) => xs.reduce((acc, x) => acc * BigInt(x), 1n);
const uniq = (arr) => Array.from(new Set(arr));
const EINSUM_COMPONENT_RE = /\p{ID_Start}|\.\.\./gu;
const einsumParseCache = /* @__PURE__ */ new Map();
/** Parse an Einstein notation string into a runnable `EinsumInput`. */
function parseEinsumExpression(expr, shapes) {
	return require_backend.runWithCache(einsumParseCache, [expr, shapes], () => {
		const idents = [...expr.split("->")[0].matchAll(EINSUM_COMPONENT_RE).map((m) => m[0]).filter((c) => c !== "...")];
		if (!expr.includes("->")) {
			const counts = /* @__PURE__ */ new Map();
			for (const c of idents) counts.set(c, (counts.get(c) ?? 0) + 1);
			const outputIndices = Array.from(counts.entries()).filter(([, count]) => count === 1).map(([char]) => char).sort();
			if (expr.includes("...")) outputIndices.splice(0, 0, "...");
			expr += "->" + outputIndices.join("");
		}
		const identToIndex = new Map(uniq(idents).sort().map((c, i) => [c, i]));
		const componentsToIndices = (components, rank) => components.flatMap((c) => {
			if (c === "...") {
				const start = rank !== void 0 ? components.length - 1 + ellipsisRank - rank : 0;
				return require_backend.range(identToIndex.size + start, identToIndex.size + ellipsisRank);
			}
			return identToIndex.get(c);
		});
		let ellipsisRank = 0;
		const [lhs, rhs] = expr.split("->");
		const lhsComponents = lhs.split(",").map((part) => [...part.matchAll(EINSUM_COMPONENT_RE).map((m) => m[0])]);
		const rhsComponents = [...rhs.matchAll(EINSUM_COMPONENT_RE)].map((m) => m[0]);
		for (const [i, components] of lhsComponents.entries()) {
			const shape$1 = shapes[i];
			const ellipsisIndex = components.indexOf("...");
			if (ellipsisIndex !== -1) {
				if (components.lastIndexOf("...") !== ellipsisIndex) throw new Error("Multiple ellipses in one einsum operand is not allowed");
				const numExplicit = components.length - 1;
				if (shape$1.length < numExplicit) throw new Error(`Einsum operand ${i} has shape ${JSON.stringify(shape$1)} but indexed with "${components.join("")}"`);
				ellipsisRank = Math.max(ellipsisRank, shape$1.length - numExplicit);
			}
		}
		const lhsIndices = lhsComponents.map((components, i) => componentsToIndices(components, shapes[i].length));
		const rhsIndex = componentsToIndices(rhsComponents);
		return {
			shapes,
			lhsIndices,
			rhsIndex
		};
	});
}
var EinsumPath = class {
	/** Parsed and normalized input for the einsum. */
	input;
	/** Mapping of each index number to its size in the shape array. */
	sizeMap;
	/**
	* A list of tensor contractions.
	*
	* This is ordered by operation order. Each entry corresponds to a single
	* elementwise product and/or inner contraction between two tensors, and it
	* contains the indices of the tensors to be contracted.
	*
	* The indices of input tensors are [0..n), and each intermediate from the
	* path at index i produces a new tensor at index n + i at the end
	* (opt_einsum internally calls this "SSA form").
	*
	* Invariants:
	* - Each group in the path consists of two tensors.
	* - For n input tensors, there are n-1 groups in the path.
	* - Every tensor must be in the path exactly once, except the final output.
	*
	* @example
	* Given einsum for `(A, B, C)`, this path corresponds to `(A, B)` and then
	* `(AB, C)`.
	* ```
	* [[0, 1], [3, 2]]
	* ```
	*/
	path;
	constructor(input, sizeMap, path) {
		this.input = input;
		this.sizeMap = sizeMap;
		this.path = path;
	}
	/** Shape of the final output tensor. */
	get outputShape() {
		return this.input.rhsIndex.map((i) => this.sizeMap.get(i));
	}
	/** Estimate the number of FLOPs to execute this einsum path. */
	get approximateFlops() {
		return approximatePathFlops(this.input, this.sizeMap, this.path);
	}
};
function approximatePathFlops(input, sizeMap, path) {
	if (path.length == 0) {
		const [indices$1] = input.lhsIndices;
		return bprod(...uniq(indices$1).map((i) => sizeMap.get(i)));
	}
	const indexUsageCounts = [];
	for (const idx of [...input.lhsIndices.flat(), ...input.rhsIndex]) indexUsageCounts[idx] = (indexUsageCounts[idx] ?? 0) + 1;
	const indices = [...input.lhsIndices];
	let totalFlops = 0n;
	for (const tensorGroup of path) {
		const indexReduced = [];
		const indexGroup = [];
		for (const tensorIdx of tensorGroup) for (const idx of indices[tensorIdx]) {
			if (!indexGroup.includes(idx)) indexGroup.push(idx);
			if (--indexUsageCounts[idx] === 0) indexReduced.push(idx);
		}
		totalFlops += approximateCountFlops(indexGroup, indexReduced.length > 0, tensorGroup.length, sizeMap);
		const newIndex = indexGroup.filter((x) => !indexReduced.includes(x));
		for (const idx of newIndex) indexUsageCounts[idx]++;
		indices.push(newIndex);
	}
	return totalFlops;
}
function approximateCountFlops(indexGroup, hasReduction, numTerms, sizeMap) {
	const elements = bprod(...indexGroup.map((i) => sizeMap.get(i)));
	const flopsPerLoopIteration = BigInt(numTerms) - 1n + (hasReduction ? 1n : 0n);
	return elements * flopsPerLoopIteration;
}
/** Compute size for each index in the einsum expression, also validates input. */
function computeSizeMap({ shapes, lhsIndices, rhsIndex }) {
	if (shapes.length === 0) throw new Error("Einsum must have at least one input tensor");
	if (lhsIndices.length !== shapes.length) throw new Error(`Mismatched number of lhs operands (${lhsIndices.length}) and shapes (${shapes.length})`);
	for (let i = 0; i < shapes.length; i++) if (lhsIndices[i].length !== shapes[i].length) throw new Error(`Mismatched number of indices (${lhsIndices[i].length}) and shape (${JSON.stringify(shapes[i])}) for operand ${i}`);
	const rhsIndexSet = /* @__PURE__ */ new Set();
	for (const idx of rhsIndex) {
		if (rhsIndexSet.has(idx)) throw new Error(`Repeated index ${idx} in einsum output`);
		rhsIndexSet.add(idx);
	}
	const sizeMap = /* @__PURE__ */ new Map();
	for (let i = 0; i < shapes.length; i++) {
		const shape$1 = shapes[i];
		const lhsIndex = lhsIndices[i];
		for (let j = 0; j < lhsIndex.length; j++) {
			const idx = lhsIndex[j];
			const dim = shape$1[j];
			const existing = sizeMap.get(idx);
			if (existing === void 0 || existing === 1) sizeMap.set(idx, dim);
			else if (existing !== dim && dim !== 1) throw new Error(`Inconsistent size for index ${idx} in einsum: ${existing} vs ${dim}`);
		}
	}
	for (const [idx, size$1] of sizeMap) if (!Number.isInteger(idx) || idx < 0) throw new Error(`Invalid index ${idx} in einsum expression, must be non-negative integer`);
	else if (size$1 < 0) throw new Error(`Invalid size ${size$1} for index ${idx} in einsum expression, must be non-negative`);
	for (const idx of rhsIndex) if (!sizeMap.has(idx)) throw new Error(`Output index ${idx} not present in einsum inputs`);
	return sizeMap;
}
const einsumPathCache = /* @__PURE__ */ new Map();
function computeEinsumPath(input, method) {
	if (!method) method = input.shapes.length <= 5 ? "optimal" : "naive";
	return require_backend.runWithCache(einsumPathCache, [input, method], () => {
		const sizeMap = computeSizeMap(input);
		if (input.shapes.length === 1) return new EinsumPath(input, sizeMap, []);
		switch (method) {
			case "naive": return computePathNaive(input, sizeMap);
			case "optimal": return computePathOptimal(input, sizeMap);
			default: throw new Error(`Unknown computePath method: ${method}`);
		}
	});
}
function computePathNaive(input, sizeMap) {
	const n = input.shapes.length;
	const path = [];
	let lastTensorIndex = 0;
	for (let i = 1; i < n; i++) {
		path.push([lastTensorIndex, i]);
		lastTensorIndex = n + i - 1;
	}
	return new EinsumPath(input, sizeMap, path);
}
function computePathOptimal(input, sizeMap) {
	const n = input.shapes.length;
	let bestPath = null;
	let bestFlops = null;
	for (const path of allPaths(require_backend.range(n), n)) {
		const flops = approximatePathFlops(input, sizeMap, path);
		if (bestFlops === null || flops < bestFlops) {
			bestPath = path;
			bestFlops = flops;
		}
	}
	return new EinsumPath(input, sizeMap, bestPath);
}
function* allPaths(tensors, next) {
	if (tensors.length === 2) {
		yield [[tensors[0], tensors[1]]];
		return;
	}
	for (let i = 0; i < tensors.length; i++) for (let j = i + 1; j < tensors.length; j++) {
		const pair = [tensors[i], tensors[j]];
		const newTensors = tensors.filter((t) => t !== pair[0] && t !== pair[1]);
		newTensors.push(next);
		for (const subpath of allPaths(newTensors, next + 1)) yield [pair, ...subpath];
	}
}

//#endregion
//#region src/library/numpy-fft.ts
var numpy_fft_exports = {};
__export(numpy_fft_exports, {
	fft: () => fft,
	ifft: () => ifft
});
var import_usingCtx$6 = /* @__PURE__ */ __toESM(require_usingCtx(), 1);
function checkPairInput(name, a) {
	const fullName = `jax.numpy.fft.${name}`;
	if (!require_backend.deepEqual(a.real.shape, a.imag.shape)) throw new Error(`${fullName}: real and imaginary parts must have the same shape, got ${JSON.stringify(a.real.shape)} and ${JSON.stringify(a.imag.shape)}`);
	if (a.real.dtype !== a.imag.dtype) throw new Error(`${fullName}: real and imaginary parts must have the same dtype, got ${a.real.dtype} and ${a.imag.dtype}`);
	if (!require_backend.isFloatDtype(a.real.dtype)) throw new Error(`${fullName}: input must have a float dtype, got ${a.real.dtype}`);
}
function checkPowerOfTwo(name, n) {
	if ((n & n - 1) !== 0) throw new Error(`jax.numpy.fft.${name}: size must be a power of two, got ${n}`);
}
const fftUpdate = jit$1(function fftUpdate$1(i, { real, imag }) {
	try {
		var _usingCtx$1 = (0, import_usingCtx$6.default)();
		const half = 2 ** i;
		real = real.reshape([-1, 2 * half]);
		imag = imag.reshape([-1, 2 * half]);
		const k = arange(0, half, 1, { dtype: real.dtype });
		anonymousConstArrays.add(k);
		const theta = _usingCtx$1.u(k.mul(-Math.PI / half));
		const wr = cos(theta);
		const wi = sin(theta);
		const ur = real.slice([], [0, half]);
		const ui = imag.slice([], [0, half]);
		const vr = real.slice([], [half, 2 * half]);
		const vi = imag.slice([], [half, 2 * half]);
		const vrWr = _usingCtx$1.u(vr.mul(wr));
		const viWi = _usingCtx$1.u(vi.mul(wi));
		const tr = vrWr.sub(viWi);
		const vrWi = _usingCtx$1.u(vr.mul(wi));
		const viWr = _usingCtx$1.u(vi.mul(wr));
		const ti = vrWi.add(viWr);
		return {
			real: concatenate([ur.add(tr), ur.sub(tr)], -1),
			imag: concatenate([ui.add(ti), ui.sub(ti)], -1)
		};
	} catch (_) {
		_usingCtx$1.e = _;
	} finally {
		_usingCtx$1.d();
	}
}, { staticArgnums: [0] });
/**
* Compute a one-dimensional discrete Fourier transform.
*
* Currently, the size of the axis must be a power of two.
*/
function fft(a, axis = -1) {
	checkPairInput("fft", a);
	let { real, imag } = a;
	axis = require_backend.checkAxis(axis, real.ndim);
	const n = real.shape[axis];
	checkPowerOfTwo("fft", n);
	const logN = Math.log2(n);
	const d = [];
	try {
		let perm = null;
		if (axis !== real.ndim - 1) {
			perm = require_backend.range(real.ndim);
			perm.splice(axis, 1);
			perm.push(axis);
			real = real.transpose(perm);
			d.push(real);
			imag = imag.transpose(perm);
			d.push(imag);
		}
		const originalShape = real.shape;
		const realR = real.reshape([-1, ...require_backend.rep(logN, 2)]);
		d.push(realR);
		const realT = realR.transpose([0, ...require_backend.range(1, logN + 1).reverse()]);
		d.push(realT);
		real = realT.flatten();
		d.push(real);
		const imagR = imag.reshape([-1, ...require_backend.rep(logN, 2)]);
		d.push(imagR);
		const imagT = imagR.transpose([0, ...require_backend.range(1, logN + 1).reverse()]);
		d.push(imagT);
		imag = imagT.flatten();
		d.push(imag);
		for (let i = 0; i < logN; i++) {
			({real, imag} = fftUpdate(i, {
				real,
				imag
			}));
			d.push(real, imag);
		}
		real = real.reshape(originalShape);
		d.push(real);
		imag = imag.reshape(originalShape);
		d.push(imag);
		if (perm !== null) {
			real = real.transpose(require_backend.invertPermutation(perm));
			d.push(real);
			imag = imag.transpose(require_backend.invertPermutation(perm));
			d.push(imag);
		}
		return {
			real,
			imag
		};
	} finally {
		for (const v of d) if (v !== real && v !== imag) v[Symbol.dispose]();
	}
}
/**
* Compute a one-dimensional inverse discrete Fourier transform.
*
* Currently, the size of the axis must be a power of two.
*/
function ifft(a, axis = -1) {
	try {
		var _usingCtx3 = (0, import_usingCtx$6.default)();
		checkPairInput("ifft", a);
		const { real, imag } = a;
		axis = require_backend.checkAxis(axis, real.ndim);
		const n = real.shape[axis];
		checkPowerOfTwo("ifft", n);
		const negImag = _usingCtx3.u(imag.mul(-1));
		const result = fft({
			real,
			imag: negImag
		}, axis);
		const fftReal = _usingCtx3.u(result.real);
		const fftImag = _usingCtx3.u(result.imag);
		const negFftImag = _usingCtx3.u(fftImag.mul(-1));
		return {
			real: fftReal.div(n),
			imag: negFftImag.div(n)
		};
	} catch (_) {
		_usingCtx3.e = _;
	} finally {
		_usingCtx3.d();
	}
}

//#endregion
//#region src/library/numpy-linalg.ts
var numpy_linalg_exports = {};
__export(numpy_linalg_exports, {
	cholesky: () => cholesky,
	det: () => det,
	diagonal: () => diagonal,
	inv: () => inv,
	lstsq: () => lstsq,
	matmul: () => matmul,
	matrixPower: () => matrixPower,
	matrixTranspose: () => matrixTranspose,
	outer: () => outer,
	slogdet: () => slogdet,
	solve: () => solve,
	tensordot: () => tensordot,
	trace: () => trace,
	vecdot: () => vecdot
});
var import_usingCtx$5 = /* @__PURE__ */ __toESM(require_usingCtx(), 1);
function checkSquare(name, a) {
	if (a.ndim < 2 || a.shape[a.ndim - 1] !== a.shape[a.ndim - 2]) throw new Error(`${name}: input must be at least 2D square matrix, got ${a.aval}`);
	return a.shape[a.ndim - 1];
}
/**
* Compute the Cholesky decomposition of a (batched) positive-definite matrix.
*
* This is like `jax.lax.linalg.cholesky()`, except with an option to symmetrize
* the input matrix, which is on by default.
*/
function cholesky(a, { upper = false, symmetrizeInput = true } = {}) {
	a = fudgeArray(a);
	checkSquare("cholesky", a);
	if (symmetrizeInput) try {
		var _usingCtx$1 = (0, import_usingCtx$5.default)();
		const at = _usingCtx$1.u(matrixTranspose(a));
		const sum$1 = _usingCtx$1.u(a.add(at));
		const sym = _usingCtx$1.u(sum$1.mul(.5));
		return cholesky$1(sym, { upper });
	} catch (_) {
		_usingCtx$1.e = _;
	} finally {
		_usingCtx$1.d();
	}
	return cholesky$1(a, { upper });
}
/** Compute the determinant of a square matrix (batched). */
function det(a) {
	try {
		var _usingCtx3 = (0, import_usingCtx$5.default)();
		a = fudgeArray(a);
		const n = checkSquare("det", a);
		const luResult = lu(a);
		const lu$2 = _usingCtx3.u(luResult[0]);
		const pivots = _usingCtx3.u(luResult[1]);
		_usingCtx3.u(luResult[2]);
		const indices = _usingCtx3.u(arange(n));
		const neq = _usingCtx3.u(pivots.notEqual(indices));
		const neqInt = _usingCtx3.u(neq.astype(int32));
		const sumAxis = _usingCtx3.u(neqInt.sum(-1));
		const parity = _usingCtx3.u(sumAxis.mod(2));
		const neg2 = _usingCtx3.u(parity.mul(-2));
		const sign$1 = _usingCtx3.u(neg2.add(1));
		const diag$1 = _usingCtx3.u(lu$2.diagonal(0, -1, -2));
		const prod$2 = _usingCtx3.u(prod$1(diag$1, -1));
		return prod$2.mul(sign$1);
	} catch (_) {
		_usingCtx3.e = _;
	} finally {
		_usingCtx3.d();
	}
}
/** Compute the inverse of a square matrix (batched). */
function inv(a) {
	try {
		var _usingCtx4 = (0, import_usingCtx$5.default)();
		a = fudgeArray(a);
		const n = checkSquare("inv", a);
		const eye$1 = _usingCtx4.u(eye(n));
		return solve(a, eye$1);
	} catch (_) {
		_usingCtx4.e = _;
	} finally {
		_usingCtx4.d();
	}
}
/**
* Return the least-squares solution to a linear equation.
*
* For overdetermined systems, this finds the `x` that minimizes `norm(ax - b)`.
* For underdetermined systems, this finds the minimum-norm solution for `x`.
*
* This currently uses Cholesky decomposition to solve the normal equations,
* under the hood. The method is not as robust as QR or SVD.
*
* @param a coefficient matrix of shape `(M, N)`
* @param b right-hand side of shape `(M,)` or `(M, K)`
* @return least-squares solution of shape `(N,)` or `(N, K)`
*/
function lstsq(a, b) {
	try {
		var _usingCtx5 = (0, import_usingCtx$5.default)();
		a = fudgeArray(a);
		b = fudgeArray(b);
		if (a.ndim !== 2) throw new Error(`lstsq: 'a' must be a 2D array, got ${a.aval}`);
		const [m, n] = a.shape;
		if (b.shape[0] !== m) throw new Error(`lstsq: leading dimension of 'b' must match number of rows of 'a', got ${b.aval}`);
		const at = _usingCtx5.u(matrixTranspose(a));
		if (m <= n) try {
			var _usingCtx6 = (0, import_usingCtx$5.default)();
			const aat = _usingCtx6.u(matmul(a, at));
			const l = _usingCtx6.u(cholesky(aat, { symmetrizeInput: false }));
			const lb = _usingCtx6.u(triangularSolve(l, b, {
				leftSide: true,
				lower: true
			}));
			const llb = _usingCtx6.u(triangularSolve(l, lb, {
				leftSide: true,
				lower: true,
				transposeA: true
			}));
			return matmul(at, llb);
		} catch (_) {
			_usingCtx6.e = _;
		} finally {
			_usingCtx6.d();
		}
		else try {
			var _usingCtx7 = (0, import_usingCtx$5.default)();
			const ata = _usingCtx7.u(matmul(at, a));
			const l = _usingCtx7.u(cholesky(ata, { symmetrizeInput: false }));
			const atb = _usingCtx7.u(matmul(at, b));
			const lb = _usingCtx7.u(triangularSolve(l, atb, {
				leftSide: true,
				lower: true
			}));
			return triangularSolve(l, lb, {
				leftSide: true,
				lower: true,
				transposeA: true
			});
		} catch (_) {
			_usingCtx7.e = _;
		} finally {
			_usingCtx7.d();
		}
	} catch (_) {
		_usingCtx5.e = _;
	} finally {
		_usingCtx5.d();
	}
}
/** Raise a square matrix to an integer power, via repeated squarings. */
function matrixPower(a, n) {
	if (!Number.isInteger(n)) throw new Error(`matrixPower: exponent must be an integer, got ${n}`);
	a = fudgeArray(a);
	const m = checkSquare("matrixPower", a);
	if (n === 0) try {
		var _usingCtx8 = (0, import_usingCtx$5.default)();
		const eye$1 = _usingCtx8.u(eye(m));
		return broadcastTo(eye$1, a.shape);
	} catch (_) {
		_usingCtx8.e = _;
	} finally {
		_usingCtx8.d();
	}
	const isInputOwned = n < 0;
	if (n < 0) {
		a = inv(a);
		n = -n;
	}
	let result = null;
	let a2k = a;
	for (let k = 0; n; k++) {
		if (k > 0) {
			const prev = a2k;
			a2k = matmul(a2k, a2k);
			if (prev !== a || isInputOwned) prev[Symbol.dispose]();
		}
		if (n % 2 === 1) {
			const prev = result;
			result = result === null ? a2k : matmul(result, a2k);
			if (prev !== null && prev !== a2k) prev[Symbol.dispose]();
		}
		n = Math.floor(n / 2);
	}
	if (a2k !== result && (a2k !== a || isInputOwned)) a2k[Symbol.dispose]();
	return result;
}
/** Return sign and natural logarithm of the determinant of `a`. */
function slogdet(a) {
	try {
		var _usingCtx9 = (0, import_usingCtx$5.default)();
		a = fudgeArray(a);
		const n = checkSquare("slogdet", a);
		const luResult = lu(a);
		const lu$2 = _usingCtx9.u(luResult[0]);
		const pivots = _usingCtx9.u(luResult[1]);
		_usingCtx9.u(luResult[2]);
		const indices = _usingCtx9.u(arange(n));
		const neq = _usingCtx9.u(pivots.notEqual(indices));
		const neqInt = _usingCtx9.u(neq.astype(int32));
		const parityBase = _usingCtx9.u(neqInt.sum(-1));
		const diag$1 = _usingCtx9.u(lu$2.diagonal(0, -1, -2));
		const diagNeg = _usingCtx9.u(diag$1.less(0));
		const diagNegInt = _usingCtx9.u(diagNeg.astype(int32));
		const diagNegSum = _usingCtx9.u(diagNegInt.sum(-1));
		const parityTotal = _usingCtx9.u(parityBase.add(diagNegSum));
		const parity = _usingCtx9.u(parityTotal.mod(2));
		const neg2 = _usingCtx9.u(parity.mul(-2));
		const sign$1 = neg2.add(1);
		const absDiag = _usingCtx9.u(absolute(diag$1));
		const logDiag = _usingCtx9.u(log(absDiag));
		const logabsdet = logDiag.sum(-1);
		return [sign$1, logabsdet];
	} catch (_) {
		_usingCtx9.e = _;
	} finally {
		_usingCtx9.d();
	}
}
/**
* Solve a linear system of equations.
*
* This solves a (batched) linear system of equations `a @ x = b` for `x` given
* `a` and `b`. If `a` is singular, this will return `nan` or `inf` values.
*
* @param a - Coefficient matrix of shape `(..., N, N)`.
* @param b - Values of shape `(N,)` or `(..., N, M)`.
* @returns Solution `x` of shape `(..., N)` or `(..., N, M)`.
*/
function solve(a, b) {
	a = fudgeArray(a);
	b = fudgeArray(b);
	const n = checkSquare("solve", a);
	if (b.ndim === 0) throw new Error(`solve: b cannot be scalar`);
	const bIs1d = b.ndim === 1;
	const d = [];
	try {
		if (bIs1d) {
			b = b.reshape([...b.shape, 1]);
			d.push(b);
		}
		if (b.shape[b.ndim - 2] !== n) throw new Error(`solve: leading dimension of b must match size of a, got a=${a.aval}, b=${b.aval}`);
		const m = b.shape[b.ndim - 1];
		const batchDims = require_backend.generalBroadcast(a.shape.slice(0, -2), b.shape.slice(0, -2));
		a = broadcastTo(a, [
			...batchDims,
			n,
			n
		]);
		d.push(a);
		b = broadcastTo(b, [
			...batchDims,
			n,
			m
		]);
		d.push(b);
		const [lu$2, pivots, permutation] = lu(a);
		d.push(lu$2, pivots, permutation);
		const arangeN = arange(n);
		d.push(arangeN);
		const permR = permutation.reshape([...permutation.shape, 1]);
		d.push(permR);
		const eq = arangeN.equal(permR);
		d.push(eq);
		const P = eq.astype(b.dtype);
		d.push(P);
		const Pb = matmul(P, b);
		d.push(Pb);
		const LPb = triangularSolve(lu$2, Pb, {
			leftSide: true,
			lower: true,
			unitDiagonal: true
		});
		d.push(LPb);
		let x = triangularSolve(lu$2, LPb, {
			leftSide: true,
			lower: false
		});
		if (bIs1d) {
			d.push(x);
			x = squeeze(x, -1);
		}
		return x;
	} finally {
		for (const v of d) v[Symbol.dispose]();
	}
}

//#endregion
//#region src/library/numpy/dtype-info.ts
/** Machine limits for floating-point types. */
function finfo(dtype) {
	if (!require_backend.isFloatDtype(dtype)) throw new Error(`finfo: received ${dtype}, must be a floating-point type`);
	switch (dtype) {
		case require_backend.DType.Float16: return Object.freeze({
			bits: 16,
			dtype: require_backend.DType.Float16,
			eps: 2 ** -10,
			epsneg: 2 ** -11,
			machep: -10,
			max: 65504,
			maxexp: 16,
			min: -65504,
			minexp: -14,
			negep: -24,
			nexp: 5,
			nmant: 10,
			precision: 3,
			resolution: .001,
			smallestNormal: 2 ** -14,
			smallestSubnormal: 2 ** -24
		});
		case require_backend.DType.Float32: return Object.freeze({
			bits: 32,
			dtype: require_backend.DType.Float32,
			eps: 2 ** -23,
			epsneg: 2 ** -24,
			machep: -23,
			max: 34028234663852886e22,
			maxexp: 128,
			min: -34028234663852886e22,
			minexp: -126,
			negep: -24,
			nexp: 8,
			nmant: 23,
			precision: 6,
			resolution: 1e-6,
			smallestNormal: 2 ** -126,
			smallestSubnormal: 2 ** -149
		});
		case require_backend.DType.Float64: return Object.freeze({
			bits: 64,
			dtype: require_backend.DType.Float64,
			eps: 2 ** -52,
			epsneg: 2 ** -53,
			machep: -52,
			max: Number.MAX_VALUE,
			maxexp: 1024,
			min: -Number.MAX_VALUE,
			minexp: -1022,
			negep: -53,
			nexp: 11,
			nmant: 52,
			precision: 15,
			resolution: 1e-15,
			smallestNormal: 2 ** -1022,
			smallestSubnormal: 2 ** -1074
		});
		default: throw new Error(`finfo: unsupported dtype ${dtype}`);
	}
}
/** Machine limits for integer types. */
function iinfo(dtype) {
	switch (dtype) {
		case require_backend.DType.Int32: return Object.freeze({
			bits: 32,
			dtype: require_backend.DType.Int32,
			max: 2147483647,
			min: -2147483648
		});
		case require_backend.DType.Uint32: return Object.freeze({
			bits: 32,
			dtype: require_backend.DType.Uint32,
			max: 4294967295,
			min: 0
		});
		default: throw new Error(`iinfo: unsupported dtype ${dtype}`);
	}
}

//#endregion
//#region src/library/numpy.ts
var numpy_exports = {};
__export(numpy_exports, {
	Array: () => Array$1,
	DType: () => require_backend.DType,
	abs: () => absolute,
	absolute: () => absolute,
	acos: () => acos,
	acosh: () => arccosh,
	add: () => add,
	all: () => all,
	allclose: () => allclose,
	any: () => any,
	arange: () => arange,
	arccos: () => acos,
	arccosh: () => arccosh,
	arcsin: () => asin,
	arcsinh: () => arcsinh,
	arctan: () => atan,
	arctan2: () => atan2,
	arctanh: () => arctanh,
	argmax: () => argmax,
	argmin: () => argmin,
	argsort: () => argsort,
	array: () => array,
	asin: () => asin,
	asinh: () => arcsinh,
	astype: () => astype,
	atan: () => atan,
	atan2: () => atan2,
	atanh: () => arctanh,
	bool: () => bool,
	broadcastArrays: () => broadcastArrays,
	broadcastShapes: () => broadcastShapes,
	broadcastTo: () => broadcastTo,
	cbrt: () => cbrt,
	ceil: () => ceil,
	clip: () => clip,
	columnStack: () => columnStack,
	concatenate: () => concatenate,
	convolve: () => convolve,
	corrcoef: () => corrcoef,
	correlate: () => correlate,
	cos: () => cos,
	cosh: () => cosh,
	cov: () => cov,
	cumsum: () => cumsum,
	cumulativeSum: () => cumsum,
	deg2rad: () => deg2rad,
	degrees: () => degrees,
	diag: () => diag,
	diagonal: () => diagonal,
	divide: () => trueDivide,
	divmod: () => divmod,
	dot: () => dot$1,
	dstack: () => dstack,
	e: () => e,
	einsum: () => einsum,
	equal: () => equal,
	eulerGamma: () => eulerGamma,
	exp: () => exp,
	exp2: () => exp2,
	expandDims: () => expandDims,
	expm1: () => expm1,
	eye: () => eye,
	fft: () => numpy_fft_exports,
	finfo: () => finfo,
	flip: () => flip,
	fliplr: () => fliplr,
	flipud: () => flipud,
	float16: () => float16,
	float32: () => float32,
	float64: () => float64,
	floor: () => floor,
	floorDivide: () => floorDivide,
	fmod: () => fmod,
	frexp: () => frexp,
	full: () => full,
	fullLike: () => fullLike$1,
	greater: () => greater,
	greaterEqual: () => greaterEqual,
	hamming: () => hamming,
	hann: () => hann,
	heaviside: () => heaviside,
	hstack: () => hstack,
	hypot: () => hypot,
	identity: () => identity$1,
	iinfo: () => iinfo,
	inf: () => inf,
	inner: () => inner,
	int32: () => int32,
	isfinite: () => isfinite,
	isinf: () => isinf,
	isnan: () => isnan,
	isneginf: () => isneginf,
	isposinf: () => isposinf,
	ldexp: () => ldexp,
	less: () => less,
	lessEqual: () => lessEqual,
	linalg: () => numpy_linalg_exports,
	linspace: () => linspace,
	log: () => log,
	log10: () => log10,
	log1p: () => log1p,
	log2: () => log2,
	logspace: () => logspace,
	matmul: () => matmul,
	matrixTranspose: () => matrixTranspose,
	max: () => max,
	maximum: () => maximum,
	mean: () => mean,
	meshgrid: () => meshgrid,
	min: () => min,
	minimum: () => minimum,
	moveaxis: () => moveaxis$1,
	multiply: () => multiply,
	nan: () => nan,
	nanToNum: () => nanToNum,
	ndim: () => ndim,
	negative: () => negative,
	notEqual: () => notEqual,
	ones: () => ones,
	onesLike: () => onesLike,
	outer: () => outer,
	pad: () => pad,
	permuteDims: () => transpose,
	pi: () => pi,
	positive: () => positive,
	pow: () => power,
	power: () => power,
	prod: () => prod$1,
	promoteTypes: () => require_backend.promoteTypes,
	ptp: () => ptp,
	rad2deg: () => rad2deg,
	radians: () => radians,
	ravel: () => ravel,
	reciprocal: () => reciprocal,
	remainder: () => remainder,
	repeat: () => repeat,
	reshape: () => reshape,
	shape: () => shape,
	sign: () => sign,
	sin: () => sin,
	sinc: () => sinc,
	sinh: () => sinh,
	size: () => size,
	sort: () => sort,
	split: () => split$1,
	sqrt: () => sqrt,
	square: () => square,
	squeeze: () => squeeze,
	stack: () => stack,
	std: () => std,
	subtract: () => subtract,
	sum: () => sum,
	swapaxes: () => swapaxes,
	take: () => take,
	tan: () => tan,
	tanh: () => tanh,
	tensordot: () => tensordot,
	tile: () => tile,
	trace: () => trace,
	transpose: () => transpose,
	tri: () => tri,
	tril: () => tril,
	triu: () => triu,
	trueDivide: () => trueDivide,
	trunc: () => trunc,
	uint32: () => uint32,
	var_: () => var_,
	vdot: () => vdot,
	vecdot: () => vecdot,
	vstack: () => vstack,
	where: () => where,
	zeros: () => zeros,
	zerosLike: () => zerosLike
});
var import_usingCtx$4 = /* @__PURE__ */ __toESM(require_usingCtx(), 1);
const float32 = require_backend.DType.Float32;
const int32 = require_backend.DType.Int32;
const uint32 = require_backend.DType.Uint32;
const bool = require_backend.DType.Bool;
const float16 = require_backend.DType.Float16;
const float64 = require_backend.DType.Float64;
/** Euler's constant, `e = 2.7182818284590...` */
const e = Math.E;
/** Euler-Mascheroni constant, `γ = 0.5772156649...` */
const eulerGamma = .5772156649015329;
/** Positive infinity. */
const inf = Number.POSITIVE_INFINITY;
/** Floating-point representation of NaN. */
const nan = NaN;
/** This is Pi, `π = 3.14159265358979...` */
const pi = Math.PI;
/** @function Element-wise addition, with broadcasting. */
const add = add$1;
/** @function Element-wise multiplication, with broadcasting. */
const multiply = mul;
/** @function Numerical negative of every element of an array. */
const negative = neg;
/** @function Calculate element-wise reciprocal of the input. This is `1/x`. */
const reciprocal = reciprocal$1;
/** @function Round input down to the nearest integer. */
const floor = floor$1;
/** @function Round input up to the nearest integer. */
const ceil = ceil$1;
/** @function Element-wise sine function (takes radians). */
const sin = sin$1;
/** @function Element-wise cosine function (takes radians). */
const cos = cos$1;
/** @function Element-wise inverse sine function (inverse of sin). */
const asin = asin$1;
/** @function Element-wise inverse tangent function (inverse of tan). */
const atan = atan$1;
/** @function Calculate the exponential of all elements in the input array. */
const exp = exp$1;
/** @function Calculate the natural logarithm of all elements in the input array. */
const log = log$1;
/** @function Calculate the square root of all elements in the input array. */
const sqrt = sqrt$1;
/** @function Return element-wise minimum of the input arrays. */
const minimum = min$1;
/** @function Return element-wise maximum of the input arrays. */
const maximum = max$1;
/** @function Compare two arrays element-wise. */
const greater = greater$1;
/** @function Compare two arrays element-wise. */
const less = less$1;
/** @function Compare two arrays element-wise. */
const equal = equal$1;
/** @function Compare two arrays element-wise. */
const notEqual = notEqual$1;
/** @function Compare two arrays element-wise. */
const greaterEqual = greaterEqual$1;
/** @function Compare two arrays element-wise. */
const lessEqual = lessEqual$1;
/** @function Element-wise ternary operator, evaluates to `x` if cond else `y`. */
const where = where$1;
/**
* @function
* Permute the dimensions of an array. Defaults to reversing the axis order.
*/
const transpose = transpose$1;
/**
* @function
* Give a new shape to an array without changing its data.
*
* One shape dimension can be -1. In this case, the value is inferred from the
* length of the array and remaining dimensions.
*/
const reshape = reshape$1;
/**
* @function
* Move axes of an array to new positions. Other axes retain original order.
*/
const moveaxis$1 = moveaxis;
/**
* @function
* Add padding (zeros) to an array.
*
* The `width` argument is either an integer or pair of integers, in which case
* all axes are padded with the same width. Or if it is an array of pairs, each
* pair specifies the padding for its corresponding axis.
*/
const pad = pad$1;
/**
* @function
* Return the number of dimensions of an array. Does not consume array reference.
*/
const ndim = ndim$1;
/** @function Return the shape of an array. Does not consume array reference. */
const shape = getShape;
/**
* @function
* Return an array of zeros with the same shape and type as a given array.
*/
const zerosLike = zerosLike$1;
/**
* @function
* Return an array of ones with the same shape and type as a given array.
*/
const onesLike = onesLike$1;
/**
* @function
* Return a full array with the same shape and type as a given array.
*/
const fullLike$1 = fullLike;
/**
* Return the number of elements in an array, optionally along an axis.
* Does not consume array reference.
*/
function size(a, axis) {
	const s = shape(a);
	return axis === void 0 ? require_backend.prod(s) : s[axis];
}
/** Convert an array to a specified dtype. */
function astype(a, dtype) {
	return fudgeArray(a).astype(dtype);
}
/** Sum of the elements of the array over a given axis, or axes. */
function sum(a, axis = null, opts) {
	return reduce(a, require_backend.AluOp.Add, axis, opts);
}
/** Product of the array elements over a given axis. */
function prod$1(a, axis = null, opts) {
	return reduce(a, require_backend.AluOp.Mul, axis, opts);
}
/** Return the minimum of array elements along a given axis. */
function min(a, axis = null, opts) {
	return reduce(a, require_backend.AluOp.Min, axis, opts);
}
/** Return the maximum of array elements along a given axis. */
function max(a, axis = null, opts) {
	return reduce(a, require_backend.AluOp.Max, axis, opts);
}
/**
* Test whether any array element along a given axis evaluates to True.
*
* Returns a boolean array with the same shape as `a` with the specified axis
* removed. If axis is None, returns a scalar.
*/
function any(a, axis = null, opts) {
	return fudgeArray(a).any(axis, opts);
}
/**
* Test whether all array elements along a given axis evaluate to True.
*
* Returns a boolean array with the same shape as `a` with the specified axis
* removed. If axis is None, returns a scalar.
*/
function all(a, axis = null, opts) {
	return fudgeArray(a).all(axis, opts);
}
/** Return the peak-to-peak range along a given axis (`max - min`). */
function ptp(a, axis = null, opts) {
	try {
		var _usingCtx$1 = (0, import_usingCtx$4.default)();
		a = fudgeArray(a);
		const maxVal = _usingCtx$1.u(max(a, axis, opts));
		const minVal = _usingCtx$1.u(min(a, axis, opts));
		return maxVal.sub(minVal);
	} catch (_) {
		_usingCtx$1.e = _;
	} finally {
		_usingCtx$1.d();
	}
}
/** Compute the average of the array elements along the specified axis. */
function mean(a, axis = null, opts) {
	return fudgeArray(a).mean(axis, opts);
}
/**
* Returns the indices of the minimum values along an axis.
*
* By default, index is into the flatted array, otherwise it is along the
* specified axis.
*/
function argmin(a, axis, opts) {
	a = fudgeArray(a);
	let ravelled;
	if (axis === void 0) {
		ravelled = a.ravel();
		a = ravelled;
		axis = 0;
	} else axis = require_backend.checkAxis(axis, a.ndim);
	try {
		try {
			var _usingCtx3 = (0, import_usingCtx$4.default)();
			const shape$1 = a.shape;
			const minVal = _usingCtx3.u(min(a, axis, { keepdims: true }));
			const isMax = _usingCtx3.u(equal(a, minVal));
			const length = _usingCtx3.u(array(shape$1[axis], {
				dtype: int32,
				device: a.device
			}));
			const range$1 = _usingCtx3.u(arange(shape$1[axis], 0, -1, {
				dtype: int32,
				device: a.device
			}));
			const reshaped = _usingCtx3.u(range$1.reshape([shape$1[axis], ...require_backend.rep(shape$1.length - axis - 1, 1)]));
			const isMaxInt = _usingCtx3.u(isMax.astype(require_backend.DType.Int32));
			const idx = _usingCtx3.u(isMaxInt.mul(reshaped));
			const maxIdx = _usingCtx3.u(max(idx, axis, opts));
			return length.sub(maxIdx);
		} catch (_) {
			_usingCtx3.e = _;
		} finally {
			_usingCtx3.d();
		}
	} finally {
		ravelled?.dispose();
	}
}
/**
* Returns the indices of the maximum values along an axis.
*
* By default, index is into the flatted array, otherwise it is along the
* specified axis.
*/
function argmax(a, axis, opts) {
	a = fudgeArray(a);
	let ravelled;
	if (axis === void 0) {
		ravelled = a.ravel();
		a = ravelled;
		axis = 0;
	} else axis = require_backend.checkAxis(axis, a.ndim);
	try {
		try {
			var _usingCtx4 = (0, import_usingCtx$4.default)();
			const shape$1 = a.shape;
			const maxVal = _usingCtx4.u(max(a, axis, { keepdims: true }));
			const isMax = _usingCtx4.u(equal(a, maxVal));
			const length = _usingCtx4.u(array(shape$1[axis], {
				dtype: int32,
				device: a.device
			}));
			const range$1 = _usingCtx4.u(arange(shape$1[axis], 0, -1, {
				dtype: int32,
				device: a.device
			}));
			const reshaped = _usingCtx4.u(range$1.reshape([shape$1[axis], ...require_backend.rep(shape$1.length - axis - 1, 1)]));
			const isMaxInt = _usingCtx4.u(isMax.astype(require_backend.DType.Int32));
			const idx = _usingCtx4.u(isMaxInt.mul(reshaped));
			const maxIdx = _usingCtx4.u(max(idx, axis, opts));
			return length.sub(maxIdx);
		} catch (_) {
			_usingCtx4.e = _;
		} finally {
			_usingCtx4.d();
		}
	} finally {
		ravelled?.dispose();
	}
}
/**
* Cumulative sum of elements along an axis.
*
* Currently this function is `O(n^2)`, we'll improve this later on with a
* two-phase parallel reduction algorithm.
*/
function cumsum(a, axis) {
	a = fudgeArray(a);
	const inputA = a;
	if (axis === void 0) {
		a = a.ravel();
		axis = 0;
	} else axis = require_backend.checkAxis(axis, a.ndim);
	const n = a.shape[axis];
	const a1 = moveaxis$1(a, axis, -1);
	try {
		try {
			var _usingCtx5 = (0, import_usingCtx$4.default)();
			const a2 = _usingCtx5.u(broadcast(a1, a1.shape.concat(n), [-2]));
			const trilA = _usingCtx5.u(tril(a2));
			const summed = trilA.sum(-1);
			const result = moveaxis$1(summed, -1, axis);
			if (result !== summed) summed.dispose();
			return result;
		} catch (_) {
			_usingCtx5.e = _;
		} finally {
			_usingCtx5.d();
		}
	} finally {
		if (a1 !== a) a1.dispose();
		if (a !== inputA) a.dispose();
	}
}
/** Reverse the elements in an array along the given axes. */
function flip(x, axis = null) {
	const nd = ndim(x);
	axis = require_backend.normalizeAxis(axis, nd);
	return flip$1(x, axis);
}
/**
* Split an array into multiple sub-arrays along an axis.
*
* @param a - The input array to split.
* @param indicesOrSections - If an integer, it indicates the number of equal
* sections to create along the specified axis. If a list of integers, it
* specifies the indices at which to split the array.
* @param axis - The axis along which to split the array. Default is 0.
*/
function split$1(a, indicesOrSections, axis = 0) {
	a = fudgeArray(a);
	axis = require_backend.checkAxis(axis, a.ndim);
	const size$1 = a.shape[axis];
	let sizes;
	if (typeof indicesOrSections === "number") {
		if (size$1 % indicesOrSections !== 0) throw new Error(`Array of size ${size$1} cannot be split into ${indicesOrSections} equal parts`);
		const partSize = size$1 / indicesOrSections;
		sizes = require_backend.rep(indicesOrSections, partSize);
	} else {
		const indices = indicesOrSections.map((i) => i < 0 ? i + size$1 : i);
		sizes = [indices[0]];
		for (let i = 1; i < indices.length; i++) sizes.push(indices[i] - indices[i - 1]);
		sizes.push(size$1 - indices[indices.length - 1]);
	}
	const results = [];
	for (let i = 0; i < sizes.length; i += 7) if (i === sizes.length) {
		results.push(a);
		break;
	} else if (i + 8 >= sizes.length) {
		results.push(...split$2(a, axis, sizes.slice(i)));
		break;
	} else {
		const groupSizes = [...sizes.slice(i, i + 7), sizes.slice(i + 7).reduce((x, y) => x + y, 0)];
		const outs = split$2(a, axis, groupSizes);
		results.push(...outs.slice(0, -1));
		a = outs[outs.length - 1];
	}
	return results;
}
/**
* Join a sequence of arrays along an existing axis.
*
* The arrays must have the same shape, except in the dimension corresponding to
* `axis` (the first, by default).
*
* No scalars can be passed to this function, as the axis is then ambiguous.
*/
function concatenate(xs, axis = 0) {
	if (xs.length === 0) throw new Error("Need at least one array to concatenate");
	const shapes = xs.map(shape);
	axis = require_backend.checkAxis(axis, shapes[0].length);
	for (let i = 1; i < shapes.length; i++) if (shapes[i].length !== shapes[0].length || !shapes[i].every((d, j) => j === axis || d === shapes[0][j])) throw new Error(`Cannot concatenate arrays ${xs[0].aval} and ${xs[i].aval} along axis ${axis}`);
	let result = xs[0];
	for (let i = 1; i < xs.length; i += 7) {
		const group = xs.slice(i, i + 7);
		const prev = result;
		result = concatenate$1([result, ...group], axis);
		if (prev !== xs[0]) prev.dispose();
	}
	return result;
}
/**
* Join a sequence of arrays along a new axis.
*
* The `axis` parameter specifies the index of the new axis in the dimensions of
* the result. For example, if `axis=0` it will be the first dimension and if
* `axis=-1` it will be the last dimension.
*
* All shapes must have the same shape.
*/
function stack(xs, axis = 0) {
	if (xs.length === 0) throw new Error("Need at least one array to stack");
	const shapes = xs.map((x) => shape(x));
	if (!shapes.every((s) => require_backend.deepEqual(s, shapes[0]))) throw new Error(`Cannot stack arrays with different shapes: ${JSON.stringify(shapes)}`);
	axis = require_backend.checkAxis(axis, shapes[0].length + 1);
	const newShape = shapes[0].toSpliced(axis, 0, 1);
	const fudged = xs.map((x) => fudgeArray(x));
	const newlyCreated = fudged.filter((f, i) => f !== xs[i]);
	const newArrays = fudged.map((x) => x.reshape(newShape));
	try {
		return concatenate(newArrays, axis);
	} finally {
		for (const a of newArrays) a[Symbol.dispose]();
		for (const a of newlyCreated) a[Symbol.dispose]();
	}
}
/**
* Horizontally stack arrays. Inputs are promoted to rank at least 1, then
* concatenated along axis 1 (if rank-2 or higher) or 0 (if rank-1).
*/
function hstack(xs) {
	if (xs.length === 0) throw new Error("Need at least one array to hstack");
	const nds = xs.map(ndim);
	if (nds.some((n) => n !== nds[0])) throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
	if (nds[0] === 0) return stack(xs);
	else if (nds[0] === 1) return concatenate(xs);
	else return concatenate(xs, 1);
}
/**
* Vertically stack arrays. Inputs are promoted to rank at least 2, then
* concatenated along axis 0.
*/
function vstack(xs) {
	if (xs.length === 0) throw new Error("Need at least one array to vstack");
	const nds = xs.map(ndim);
	if (nds.some((n) => n !== nds[0])) throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
	if (nds[0] === 0) try {
		var _usingCtx6 = (0, import_usingCtx$4.default)();
		const stacked = _usingCtx6.u(stack(xs));
		return stacked.reshape([-1, 1]);
	} catch (_) {
		_usingCtx6.e = _;
	} finally {
		_usingCtx6.d();
	}
	else if (nds[0] === 1) return stack(xs);
	else return concatenate(xs);
}
/**
* Stack arrays depth-wise. Inputs are promoted to rank at least 3, then
* concatenated along axis 2.
*/
function dstack(xs) {
	if (xs.length === 0) throw new Error("Need at least one array to dstack");
	const nds = xs.map(ndim);
	if (nds.some((n) => n !== nds[0])) throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
	if (nds[0] === 0) try {
		var _usingCtx7 = (0, import_usingCtx$4.default)();
		const stacked = _usingCtx7.u(stack(xs));
		return stacked.reshape([
			1,
			1,
			-1
		]);
	} catch (_) {
		_usingCtx7.e = _;
	} finally {
		_usingCtx7.d();
	}
	else if (nds[0] === 1) try {
		var _usingCtx8 = (0, import_usingCtx$4.default)();
		const ret = _usingCtx8.u(stack(xs, -1));
		return ret.reshape([1, ...ret.shape]);
	} catch (_) {
		_usingCtx8.e = _;
	} finally {
		_usingCtx8.d();
	}
	else if (nds[0] === 2) return stack(xs, -1);
	else return concatenate(xs, 2);
}
/**
* Stack arrays column-wise. Inputs are promoted to rank at least 2, then
* concatenated along axis 1.
*/
function columnStack(xs) {
	if (xs.length === 0) throw new Error("Need at least one array to columnStack");
	const nds = xs.map(ndim);
	if (nds.some((n) => n !== nds[0])) throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
	if (nds[0] === 0) try {
		var _usingCtx9 = (0, import_usingCtx$4.default)();
		const stacked = _usingCtx9.u(stack(xs));
		return stacked.reshape([1, -1]);
	} catch (_) {
		_usingCtx9.e = _;
	} finally {
		_usingCtx9.d();
	}
	else if (nds[0] === 1) return stack(xs, -1);
	else return concatenate(xs, 1);
}
/** Flip an array vertically (axis=0). */
function flipud(x) {
	return flip(x, 0);
}
/** Flip an array horizontally (axis=1). */
function fliplr(x) {
	return flip(x, 1);
}
/** Interchange two axes of an array. */
function swapaxes(a, axis1, axis2) {
	a = fudgeArray(a);
	axis1 = require_backend.checkAxis(axis1, a.ndim);
	axis2 = require_backend.checkAxis(axis2, a.ndim);
	if (axis1 === axis2) return a;
	const perm = require_backend.range(a.ndim);
	perm[axis1] = axis2;
	perm[axis2] = axis1;
	return transpose(a, perm);
}
/** Transpose the last two dimensions of an array. */
function matrixTranspose(a) {
	if (ndim(a) < 2) throw new Error(`matrixTranspose: input array must be at least 2D`);
	return moveaxis$1(a, -1, -2);
}
/** Return a 1-D flattened array containing the elements of the input. */
function ravel(a) {
	return fudgeArray(a).ravel();
}
/** Remove one or more length-1 axes from an array. */
function squeeze(a, axis = null) {
	const as = shape(a);
	if (axis === null) axis = require_backend.range(as.length).filter((i) => as[i] === 1);
	else if (typeof axis === "number") axis = [axis];
	axis = axis.map((a$1) => require_backend.checkAxis(a$1, as.length));
	for (const a$1 of axis) if (as[a$1] !== 1) throw new Error("Cannot squeeze axis with size != 1");
	const newShape = as.filter((_, i) => !axis.includes(i));
	return reshape(a, newShape);
}
/**
* Expand the shape of an array by inserting new axes of length 1.
*
* @param a - Input array.
* @param axis - Position(s) in the expanded axes where the new axis (or axes)
*   is placed. Can be a single integer or an array of integers.
* @returns Array with the number of dimensions increased.
*
* @example
* ```ts
* const x = np.array([1, 2]);
* np.expandDims(x, 0); // Shape [1, 2]
* np.expandDims(x, 1); // Shape [2, 1]
* np.expandDims(x, [0, 2]); // Shape [1, 2, 1]
* ```
*/
function expandDims(a, axis) {
	const as = shape(a);
	axis = typeof axis === "number" ? [axis] : axis;
	axis = require_backend.normalizeAxis(axis, as.length + axis.length);
	const newShape = [];
	let srcIdx = 0;
	for (let i = 0; i < as.length + axis.length; i++) if (axis.includes(i)) newShape.push(1);
	else newShape.push(as[srcIdx++]);
	return reshape(a, newShape);
}
/**
* Repeat each element of an array after themselves.
*
* If no axis is provided, use the flattened input array, and return a flat
* output array.
*/
function repeat(a, repeats, axis) {
	try {
		var _usingCtx10 = (0, import_usingCtx$4.default)();
		if (!Number.isInteger(repeats) || repeats < 0) throw new Error(`repeat: repeats must be a non-negative integer, got ${repeats}`);
		a = fudgeArray(a);
		let ravelled;
		if (axis === void 0) {
			ravelled = ravel(a);
			a = ravelled;
			axis = 0;
		}
		axis = require_backend.checkAxis(axis, a.ndim);
		if (repeats === 1) {
			ravelled?.dispose();
			return a.reshape(a.shape);
		}
		const broadcastedShape = a.shape.toSpliced(axis + 1, 0, repeats);
		const finalShape = a.shape.toSpliced(axis, 1, a.shape[axis] * repeats);
		const broadcasted = _usingCtx10.u(broadcast(a, broadcastedShape, [axis + 1]));
		const result = broadcasted.reshape(finalShape);
		ravelled?.dispose();
		return result;
	} catch (_) {
		_usingCtx10.e = _;
	} finally {
		_usingCtx10.d();
	}
}
/**
* Construct an array by repeating A the number of times given by reps.
*
* If `A` is an array of shape `(d1, d2, ..., dn)` and `reps` is a sequence of
* integers, the resulting array will have a shape of `(reps[0] * d1,
* reps[1] * d2, ..., reps[n] * dn)`, with `A` tiled along each dimension.
*/
function tile(a, reps) {
	try {
		var _usingCtx11 = (0, import_usingCtx$4.default)();
		a = fudgeArray(a);
		if (typeof reps === "number") reps = [reps];
		if (!reps.every((r) => Number.isInteger(r) && r >= 0)) throw new Error(`tile: reps must be non-negative integers, got ${JSON.stringify(reps)}`);
		const ndiff = reps.length - a.ndim;
		let reshaped;
		if (ndiff > 0) {
			reshaped = a.reshape([...require_backend.rep(ndiff, 1), ...a.shape]);
			a = reshaped;
		}
		if (ndiff < 0) reps = [...require_backend.rep(-ndiff, 1), ...reps];
		const broadcastedShape = [];
		const broadcastAxes = [];
		for (let i = 0; i < a.ndim; i++) {
			if (reps[i] > 1) {
				broadcastedShape.push(reps[i]);
				broadcastAxes.push(broadcastedShape.length - 1);
			}
			broadcastedShape.push(a.shape[i]);
		}
		const finalShape = a.shape.map((d, i) => reps[i] * d);
		const broadcasted = _usingCtx11.u(broadcast(a, broadcastedShape, broadcastAxes));
		const result = broadcasted.reshape(finalShape);
		reshaped?.dispose();
		return result;
	} catch (_) {
		_usingCtx11.e = _;
	} finally {
		_usingCtx11.d();
	}
}
/**
* Broadcast an array to a shape, with NumPy-style broadcasing rules.
*
* In other words, this lets you append axes to the left, and/or expand
* dimensions where the shape is 1.
*/
function broadcastTo(a, shape$1) {
	const nd = ndim(a);
	if (shape$1.length < nd) throw new Error(`broadcastTo: target shape ${JSON.stringify(shape$1)} has fewer dimensions than input array: ${nd}`);
	return broadcast(a, shape$1, require_backend.range(shape$1.length - nd));
}
/** Broadcast input shapes to a common output shape. */
function broadcastShapes(...shapes) {
	if (shapes.length === 0) return [];
	return shapes.reduce(require_backend.generalBroadcast);
}
/** Broadcast arrays to a common shape. */
function broadcastArrays(...arrays) {
	const shapes = arrays.map((a) => shape(a));
	const outShape = broadcastShapes(...shapes);
	return arrays.map((a) => broadcastTo(a, outShape));
}
/**
* Return specified diagonals.
*
* If a is 2D, return the diagonal of the array with the given offset. If a is
* 3D or higher, compute diagonals along the two given axes (default: 0, 1).
*
* This returns a view over the existing array. The shape of the resulting array
* is determined by removing the two axes along which the diagonal is taken,
* then appending a new axis to the right with holding the diagonals.
*/
function diagonal(a, offset, axis1, axis2) {
	return fudgeArray(a).diagonal(offset, axis1, axis2);
}
/**
* Extract a diagonal or construct a diagonal array.
*
* If v is a 2D array, return the k-th diagonal of v (as a view). If v is a 1D
* array, return a 2D array with v on the k-th diagonal.
*/
function diag(v, k = 0) {
	const a = fudgeArray(v);
	if (!Number.isInteger(k)) throw new Error(`k must be an integer, got ${k}`);
	if (a.ndim === 1) try {
		var _usingCtx12 = (0, import_usingCtx$4.default)();
		const n = a.shape[0];
		const eyeN = _usingCtx12.u(eye(n));
		const mask = _usingCtx12.u(eyeN.equal(1));
		const zeros$1 = _usingCtx12.u(zerosLike(a));
		if (k === 0) return where(mask, a, zeros$1);
		const ret = _usingCtx12.u(where(mask, a, zeros$1));
		if (k > 0) return pad(ret, [[0, k], [k, 0]]);
		else return pad(ret, [[-k, 0], [0, -k]]);
	} catch (_) {
		_usingCtx12.e = _;
	} finally {
		_usingCtx12.d();
	}
	else if (a.ndim === 2) return diagonal(a, k);
	else throw new Error("numpy.diag only supports 1D and 2D arrays");
}
/** Calculate the sum of the diagonal of an array along the given axes. */
function trace(a, offset = 0, axis1 = 0, axis2 = 1) {
	try {
		var _usingCtx13 = (0, import_usingCtx$4.default)();
		const diag$1 = _usingCtx13.u(diagonal(a, offset, axis1, axis2));
		return diag$1.sum(-1);
	} catch (_) {
		_usingCtx13.e = _;
	} finally {
		_usingCtx13.d();
	}
}
/**
* Return a sorted copy of an array.
*
* The array is sorted along a specified axis (the last by default). This may be
* an unstable sort, and it dispatches to device-specific implementation.
*/
function sort(a, axis = -1) {
	return fudgeArray(a).sort(axis);
}
/**
* Return indices that would sort an array. Unlike `sort`, this is guaranteed to
* be a stable sorting algorithm; it always returns the smaller index first in
* event of ties.
*
* Returns an array of `int32` indices.
*
* The array is sorted along a specified axis (the last by default).
*/
function argsort(a, axis = -1) {
	return fudgeArray(a).argsort(axis);
}
/**
* Take elements from an array along an axis.
*
* This is equivalent to advanced indexing with integer indices over that
* numbered axis. By default, the flattened array is used.
*/
function take(a, indices, axis = null) {
	let ravelled;
	if (axis === null) {
		ravelled = ravel(a);
		a = ravelled;
		axis = 0;
	}
	axis = require_backend.checkAxis(axis, ndim(a));
	const result = gather(a, [indices], [axis], axis);
	ravelled?.dispose();
	return result;
}
/** Return if two arrays are element-wise equal within a tolerance. */
function allclose(actual, expected, options) {
	const { rtol = 1e-5, atol = 1e-7 } = options ?? {};
	const x = array(actual);
	const y = array(expected);
	const xOwned = x !== actual;
	const yOwned = y !== expected;
	try {
		if (!require_backend.deepEqual(x.shape, y.shape)) return false;
		const xData = x.dataSync();
		const yData = y.dataSync();
		for (let i = 0; i < xData.length; i++) {
			if (isNaN(xData[i]) !== isNaN(yData[i])) return false;
			if (Math.abs(xData[i] - yData[i]) > atol + rtol * Math.abs(yData[i])) return false;
		}
		return true;
	} finally {
		if (xOwned) x.dispose();
		if (yOwned) y.dispose();
	}
}
/** Matrix product of two arrays. */
function matmul(x, y) {
	if (ndim(x) === 0 || ndim(y) === 0) throw new Error("matmul: x and y must be at least 1D");
	x = x, y = y;
	if (y.ndim === 1) return dot$2(x, y);
	const numBatchDims = Math.min(Math.max(x.ndim, 2), y.ndim) - 2;
	return dot(x, y, {
		lhsContractingDims: [-1],
		rhsContractingDims: [-2],
		lhsBatchDims: require_backend.range(-2 - numBatchDims, -2),
		rhsBatchDims: require_backend.range(-2 - numBatchDims, -2)
	});
}
/** Dot product of two arrays. */
function dot$1(x, y) {
	if (ndim(x) === 0 || ndim(y) === 0) return multiply(x, y);
	x = x, y = y;
	if (y.ndim === 1) return dot$2(x, y);
	return dot(x, y, {
		lhsContractingDims: [-1],
		rhsContractingDims: [-2]
	});
}
/**
* Compute the tensor dot product of two N-dimensional arrays.
*
* The behavior is determined by `axes`. If an integer `k`, sum over the last
* `k` axes of x and the first `k` axes of y. If a tuple, then the first array
* corresponds to the axes of x and the second to the axes of y.
*/
function tensordot(x, y, axes = 2) {
	x = fudgeArray(x);
	y = fudgeArray(y);
	if (typeof axes === "number") axes = [require_backend.range(-axes, 0), require_backend.range(axes)];
	return dot(x, y, {
		lhsContractingDims: axes[0],
		rhsContractingDims: axes[1]
	});
}
/**
* Einstein summation.
*
* This is a general API for performing tensor reductions, products,
* transpositions, and traces using Einstein notation for referring to named
* axes. See the docs for `numpy.einsum()` for more information.
*
* <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>
*
* The full einsum API is implemented, including implicit and explicit output
* indices, ellipsis broadcasting, Unicode subscripts, and an optimal path
* ordering algorithm. It lowers to one or more calls to:
*
* - `jax.lax.dot()`
* - `jax.numpy.diagonal()`
* - `jax.numpy.sum()`
* - `jax.numpy.transpose()`
*/
function einsum(...args) {
	if (args.length === 0) throw new Error("einsum: must provide at least one argument");
	let input;
	let operands = [];
	if (typeof args[0] === "string") {
		operands = args.slice(1).map(fudgeArray);
		input = parseEinsumExpression(args[0], operands.map((x) => x.shape));
	} else {
		const n = args.length >> 1;
		const shapes = [];
		const lhsIndices = [];
		for (let i = 0; i < n; i++) {
			operands.push(fudgeArray(args[2 * i]));
			shapes.push(operands[i].shape);
			lhsIndices.push(args[2 * i + 1]);
		}
		let rhsIndex;
		if (args.length % 2 === 1) rhsIndex = args[2 * n];
		else {
			const indexCount = [];
			for (const i of lhsIndices.flat()) indexCount[i] = (indexCount[i] ?? 0) + 1;
			rhsIndex = [...indexCount.entries()].filter(([_, count]) => count === 1).map(([i, _]) => i);
		}
		input = {
			lhsIndices,
			rhsIndex,
			shapes
		};
	}
	const path = computeEinsumPath(input);
	if (require_backend.DEBUG >= 3) console.info(`einsum: computed path: ${path.approximateFlops} flops`);
	const indexUsageCounts = [];
	for (const idx of [...input.lhsIndices.flat(), ...input.rhsIndex]) indexUsageCounts[idx] = (indexUsageCounts[idx] ?? 0) + 1;
	const indices = [...input.lhsIndices];
	const numUserOperands = operands.length;
	const intermediates = [];
	const processSingleTensor = (ar$1, index$1, doNotReduce = []) => {
		index$1 = index$1.slice();
		diag: while (true) {
			for (let i = 0; i < index$1.length; i++) {
				const idx = index$1[i];
				const j = index$1.indexOf(idx, i + 1);
				if (j !== -1) {
					ar$1 = diagonal(ar$1, 0, i, j);
					intermediates.push(ar$1);
					index$1.splice(j, 1);
					index$1.splice(i, 1);
					index$1.push(idx);
					continue diag;
				}
			}
			break;
		}
		for (let i = index$1.length - 1; i >= 0; i--) {
			const idx = index$1[i];
			if (indexUsageCounts[idx] === 0 && !doNotReduce.includes(idx)) {
				ar$1 = sum(ar$1, i);
				intermediates.push(ar$1);
				index$1.splice(i, 1);
			}
		}
		return [ar$1, index$1];
	};
	for (const [i, j] of path.path) {
		let indexReduced = [];
		const indexGroup = [];
		for (const idx of [...indices[i], ...indices[j]]) {
			if (!indexGroup.includes(idx)) indexGroup.push(idx);
			if (--indexUsageCounts[idx] === 0) indexReduced.push(idx);
		}
		const [a, aidx] = processSingleTensor(operands[i], indices[i], indices[j]);
		const [b, bidx] = processSingleTensor(operands[j], indices[j], indices[i]);
		indexReduced = indexReduced.filter((idx) => aidx.includes(idx));
		const indexBatch = aidx.filter((idx) => bidx.includes(idx) && !indexReduced.includes(idx));
		const result = dot(a, b, {
			lhsContractingDims: indexReduced.map((idx) => aidx.indexOf(idx)),
			rhsContractingDims: indexReduced.map((idx) => bidx.indexOf(idx)),
			lhsBatchDims: indexBatch.map((idx) => aidx.indexOf(idx)),
			rhsBatchDims: indexBatch.map((idx) => bidx.indexOf(idx))
		});
		intermediates.push(result);
		operands.push(result);
		indices.push([
			...indexBatch,
			...aidx.filter((idx) => !bidx.includes(idx)),
			...bidx.filter((idx) => !aidx.includes(idx))
		]);
		for (const idx of indices[indices.length - 1]) ++indexUsageCounts[idx];
	}
	for (const idx of indices[operands.length - 1]) --indexUsageCounts[idx];
	const [ar, index] = processSingleTensor(operands[operands.length - 1], indices[operands.length - 1]);
	const finalPerm = input.rhsIndex.map((idx) => index.indexOf(idx));
	const finalResult = ar.transpose(finalPerm);
	const keep = new Set(operands.slice(0, numUserOperands));
	keep.add(finalResult);
	for (const a of intermediates) if (!keep.has(a)) a.dispose();
	return finalResult;
}
/**
* Compute the inner product of two arrays.
*
* Unlike `jax.numpy.matmul()` or `jax.numpy.dot()`, this always performs a
* contraction on the last axis.
*
* Returned array has shape `[...x.shape[:-1], ...y.shape[:-1]]`.
*/
function inner(x, y) {
	return dot(fudgeArray(x), fudgeArray(y), {
		lhsContractingDims: [-1],
		rhsContractingDims: [-1]
	});
}
/**
* Compute the outer product of two arrays.
*
* If the input arrays are not 1D, they will be flattened. Returned array will
* be of shape `[x.size, y.size]`.
*/
function outer(x, y) {
	try {
		var _usingCtx14 = (0, import_usingCtx$4.default)();
		const rx = _usingCtx14.u(ravel(x));
		const ry = _usingCtx14.u(ravel(y));
		const rxR = _usingCtx14.u(rx.reshape([rx.shape[0], 1]));
		return multiply(rxR, ry);
	} catch (_) {
		_usingCtx14.e = _;
	} finally {
		_usingCtx14.d();
	}
}
/** Vector dot product of two arrays along a given axis. */
function vecdot(x, y, { axis } = {}) {
	try {
		var _usingCtx15 = (0, import_usingCtx$4.default)();
		const xaxis = require_backend.checkAxis(axis ?? -1, ndim(x));
		const yaxis = require_backend.checkAxis(axis ?? -1, ndim(y));
		if (shape(x)[xaxis] !== shape(y)[yaxis]) throw new Error(`vecdot: shapes ${JSON.stringify(shape(x))} and ${JSON.stringify(shape(y))} not aligned along axis ${axis}: ${shape(x)[xaxis]} != ${shape(y)[yaxis]}`);
		const xm = _usingCtx15.u(moveaxis$1(x, xaxis, -1));
		const ym = _usingCtx15.u(moveaxis$1(y, yaxis, -1));
		return dot$2(xm, ym);
	} catch (_) {
		_usingCtx15.e = _;
	} finally {
		_usingCtx15.d();
	}
}
/**
* Return the dot product of two vectors.
*
* Like vecdot() but flattens the arguments first into vectors.
*/
function vdot(x, y) {
	try {
		var _usingCtx16 = (0, import_usingCtx$4.default)();
		const rx = _usingCtx16.u(ravel(x));
		const ry = _usingCtx16.u(ravel(y));
		return dot$2(rx, ry);
	} catch (_) {
		_usingCtx16.e = _;
	} finally {
		_usingCtx16.d();
	}
}
function _convImpl(name, x, y, mode) {
	if (x.ndim !== 1 || y.ndim !== 1) throw new Error(`${name}: both inputs must be 1D arrays, got ${x.ndim}D and ${y.ndim}D`);
	let flipOutput = false;
	if (x.shape[0] < y.shape[0]) {
		[x, y] = [y, x];
		if (name === "correlate") flipOutput = true;
	}
	const d = [];
	try {
		if (name === "convolve") {
			y = flip(y);
			d.push(y);
		}
		let padding;
		if (mode === "valid") padding = "VALID";
		else if (mode === "same") padding = "SAME_LOWER";
		else if (mode === "full") padding = [[y.shape[0] - 1, y.shape[0] - 1]];
		else throw new Error(`${name}: invalid mode ${mode}, expected "full", "same", or "valid"`);
		const xs = x.slice(null, null);
		d.push(xs);
		const ys = y.slice(null, null);
		d.push(ys);
		const convResult = conv(xs, ys, [1], padding);
		d.push(convResult);
		const z = convResult.slice(0, 0);
		if (flipOutput) {
			d.push(z);
			return flip(z);
		}
		return z;
	} finally {
		for (const v of d) v[Symbol.dispose]();
	}
}
/** Convolution of two one-dimensional arrays. */
function convolve(x, y, mode = "full") {
	return _convImpl("convolve", x, y, mode);
}
/** Correlation of two one dimensional arrays. */
function correlate(x, y, mode = "valid") {
	return _convImpl("correlate", x, y, mode);
}
/**
* Return a tuple of coordinate matrices from coordinate vectors.
*
* Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector
* fields over N-D grids, given one-dimensional coordinate arrays x1, x2,…, xn.
*/
function meshgrid(xs, { indexing } = {}) {
	indexing ??= "xy";
	for (const x of xs) if (x.ndim !== 1) throw new Error(`meshgrid: all inputs must be 1D arrays, got ${x.ndim}D array`);
	if (xs.length <= 1) return xs;
	if (indexing === "xy") {
		const [a, b, ...rest] = xs;
		const [rb, ra, ...rrest] = meshgrid([
			b,
			a,
			...rest
		], { indexing: "ij" });
		return [
			ra,
			rb,
			...rrest
		];
	}
	const shape$1 = xs.map((x) => x.shape[0]);
	return xs.map((x, i) => broadcast(x, shape$1, [...require_backend.range(i), ...require_backend.range(i + 1, xs.length)]));
}
/**
* Clip (limit) the values in an array.
*
* Given an interval, values outside the interval are clipped to the interval
* edges. For example, if an interval of [0, 1] is specified, values smaller
* than 0 become 0, and values larger than 1 become 1.
*
* If either bound is undefined, it is ignored.
*/
function clip(a, min$2, max$2) {
	a = fudgeArray(a);
	if (max$2 !== void 0 && min$2 !== void 0) try {
		var _usingCtx17 = (0, import_usingCtx$4.default)();
		const clipped = _usingCtx17.u(minimum(a, max$2));
		return maximum(clipped, min$2);
	} catch (_) {
		_usingCtx17.e = _;
	} finally {
		_usingCtx17.d();
	}
	if (max$2 !== void 0) return minimum(a, max$2);
	if (min$2 !== void 0) return maximum(a, min$2);
	return a;
}
/**
* Calculate the absolute value element-wise.
*
* This is the same function as `jax.numpy.abs()`.
*/
function absolute(x) {
	try {
		var _usingCtx18 = (0, import_usingCtx$4.default)();
		x = fudgeArray(x);
		const cond = _usingCtx18.u(less(x, 0));
		const negX = _usingCtx18.u(x.mul(-1));
		return where(cond, negX, x);
	} catch (_) {
		_usingCtx18.e = _;
	} finally {
		_usingCtx18.d();
	}
}
/** Return an element-wise indication of sign of the input. */
function sign(x) {
	try {
		var _usingCtx19 = (0, import_usingCtx$4.default)();
		x = fudgeArray(x);
		const neq = _usingCtx19.u(notEqual(x, 0));
		const lt = _usingCtx19.u(less(x, 0));
		const inner$1 = _usingCtx19.u(where(lt, -1, 1));
		const result = _usingCtx19.u(where(neq, inner$1, 0));
		const isnan$1 = _usingCtx19.u(notEqual(x, x));
		return where(isnan$1, x, result);
	} catch (_) {
		_usingCtx19.e = _;
	} finally {
		_usingCtx19.d();
	}
}
/** @function Return element-wise positive values of the input (no-op). */
const positive = fudgeArray;
/**
* Return the Hamming window of size M, a taper with a weighted cosine bell.
*
* `w(n) = 0.54 - 0.46 * cos(2πn/(M-1))` for `0 <= n <= M-1`.
*/
function hamming(M) {
	try {
		var _usingCtx20 = (0, import_usingCtx$4.default)();
		const ls = _usingCtx20.u(linspace(0, 2 * Math.PI, M));
		const c = _usingCtx20.u(cos(ls));
		const scaled = _usingCtx20.u(c.mul(-.46));
		return scaled.add(.54);
	} catch (_) {
		_usingCtx20.e = _;
	} finally {
		_usingCtx20.d();
	}
}
/**
* Return the Hann window of size M, a taper with a weighted cosine bell.
*
* `w(n) = 0.5 - 0.5 * cos(2πn/(M-1))` for `0 <= n <= M-1`.
*/
function hann(M) {
	try {
		var _usingCtx21 = (0, import_usingCtx$4.default)();
		const ls = _usingCtx21.u(linspace(0, 2 * Math.PI, M));
		const c = _usingCtx21.u(cos(ls));
		const scaled = _usingCtx21.u(c.mul(-.5));
		return scaled.add(.5);
	} catch (_) {
		_usingCtx21.e = _;
	} finally {
		_usingCtx21.d();
	}
}
/**
* @function
* Compute the Heaviside step function. It is defined piecewise:
* - `heaviside(x1, x2) = 0` for `x1 < 0`,
* - `heaviside(x1, x2) = x2` for `x1 == 0`,
* - `heaviside(x1, x2) = 1` for `x1 > 0`.
*/
const heaviside = jit$1(function heaviside$1(x1, x2) {
	return where(less(x1, 0), 0, where(equal(x1, 0), x2, 1));
});
/** Calculate element-wise square of the input array. */
function square(x) {
	x = fudgeArray(x);
	return x.mul(x);
}
/** Element-wise tangent function (takes radians). */
function tan(x) {
	try {
		var _usingCtx22 = (0, import_usingCtx$4.default)();
		x = fudgeArray(x);
		const sinX = _usingCtx22.u(sin(x));
		const cosX = _usingCtx22.u(cos(x));
		return sinX.div(cosX);
	} catch (_) {
		_usingCtx22.e = _;
	} finally {
		_usingCtx22.d();
	}
}
/**
* @function
* Return the normalized sinc function.
*
* The sinc function is defined as `sin(πx) / (πx)` for `x != 0`, and `1` for `x = 0`.
* This is the normalized sinc function commonly used in signal processing.
*
* **Note:** JVP is not supported at x=0 due to discontinuous derivative. This
* requires a custom JVP rule to handle properly (see JAX implementation).
*/
const sinc = jit$1(function sinc$1(x) {
	const pix = x.mul(Math.PI);
	return where(equal(x, 0), 1, sin(pix).div(pix));
});
/** Element-wise inverse cosine function (inverse of cos). */
function acos(x) {
	try {
		var _usingCtx23 = (0, import_usingCtx$4.default)();
		const asinX = _usingCtx23.u(asin(x));
		return subtract(pi / 2, asinX);
	} catch (_) {
		_usingCtx23.e = _;
	} finally {
		_usingCtx23.d();
	}
}
/**
* @function
* Return element-wise hypotenuse for the given legs of a right triangle.
*
* In the original NumPy/JAX implementation, this function is more numerically
* stable than `sqrt(x1**2 + x2**2)`. We don't currently implement those
* stability improvements.
*/
const hypot = jit$1(function hypot$1(x1, x2) {
	return sqrt(square(x1).add(square(x2)));
});
/**
* @function
* Element-wise arc tangent of y/x with correct quadrant.
*
* Returns the angle in radians between the positive x-axis and the point (x, y).
* The result is in the range [-π, π].
*
* Uses numerically stable formulas:
* - When x >= 0: atan2(y, x) = 2 * atan(y / (sqrt(x^2 + y^2) + x))
* - When x < 0:  atan2(y, x) = 2 * atan((sqrt(x^2 + y^2) - x) / y)
*
* The output is ill-defined when both x and y are zero.
*/
const atan2 = jit$1(function atan2$1(y, x) {
	const r = sqrt(square(x).add(square(y)));
	const xNeg = less(x, 0);
	const numer = where(xNeg, r.sub(x), y);
	const denom = where(xNeg, y, r.add(x));
	return atan(numer.div(denom)).mul(2);
});
/** Element-wise subtraction, with broadcasting. */
function subtract(x, y) {
	x = fudgeArray(x);
	y = fudgeArray(y);
	return x.sub(y);
}
/** Calculates the floating-point division of x by y element-wise. */
function trueDivide(x, y) {
	x = fudgeArray(x);
	y = fudgeArray(y);
	if (!require_backend.isFloatDtype(x.dtype) && !require_backend.isFloatDtype(y.dtype)) try {
		var _usingCtx24 = (0, import_usingCtx$4.default)();
		const xf = _usingCtx24.u(x.astype(require_backend.DType.Float32));
		const yf = _usingCtx24.u(y.astype(require_backend.DType.Float32));
		return xf.div(yf);
	} catch (_) {
		_usingCtx24.e = _;
	} finally {
		_usingCtx24.d();
	}
	return x.div(y);
}
/**
* Return the largest integer smaller or equal to the division of the inputs.
*
* The result is always rounded towards negative infinity.
*
* For floating-point inputs, this is equivalent to `floor(x / y)`.
* For integer inputs, we use `(x - remainder(x, y)) / y` to handle
* negative values correctly (note: may overflow near int32 boundaries).
*
* @param x - Dividend array.
* @param y - Divisor array.
* @returns Element-wise floor division of x by y.
*/
function floorDivide(x, y) {
	const xArr = fudgeArray(x);
	const yArr = fudgeArray(y);
	try {
		try {
			var _usingCtx25 = (0, import_usingCtx$4.default)();
			if (require_backend.isFloatDtype(xArr.dtype) || require_backend.isFloatDtype(yArr.dtype)) try {
				var _usingCtx26 = (0, import_usingCtx$4.default)();
				const div = _usingCtx26.u(trueDivide(xArr, yArr));
				return floor(div);
			} catch (_) {
				_usingCtx26.e = _;
			} finally {
				_usingCtx26.d();
			}
			const rem = _usingCtx25.u(remainder(xArr, yArr));
			const diff = _usingCtx25.u(subtract(xArr, rem));
			return diff.div(yArr);
		} catch (_) {
			_usingCtx25.e = _;
		} finally {
			_usingCtx25.d();
		}
	} finally {
		if (xArr !== x) xArr.dispose();
		if (yArr !== y) yArr.dispose();
	}
}
/**
* @function
* Calculate element-wise floating-point modulo operation.
*/
const fmod = jit$1(function fmod$1(x, y) {
	return x.sub(y.mul(idiv(x, y)));
});
/**
* @function
* Calculate element-wise remainder of the division (matches sign of y).
*/
const remainder = jit$1(function remainder$1(x, y) {
	try {
		var _usingCtx27 = (0, import_usingCtx$4.default)();
		const inner$1 = _usingCtx27.u(mod(x, y));
		const shifted = _usingCtx27.u(inner$1.add(y));
		return mod(shifted, y);
	} catch (_) {
		_usingCtx27.e = _;
	} finally {
		_usingCtx27.d();
	}
});
/**
* Return element-wise quotient and remainder simultaneously.
*
* Equivalent to `[floorDivide(x, y), remainder(x, y)]`.
*
* @param x - Dividend array.
* @param y - Divisor array.
* @returns Tuple of [quotient, remainder].
*/
function divmod(x, y) {
	const xArr = fudgeArray(x);
	const yArr = fudgeArray(y);
	const result = [floorDivide(xArr, yArr), remainder(xArr, yArr)];
	if (xArr !== x) xArr.dispose();
	if (yArr !== y) yArr.dispose();
	return result;
}
/** Round input to the nearest integer towards zero. */
function trunc(x) {
	return idiv(x, 1);
}
/**
* Compute `x1 * 2 ** x2` as a standard multiplication and exponentiation.
*
* This is the inverse of `frexp()`.
*/
function ldexp(x1, x2) {
	try {
		var _usingCtx28 = (0, import_usingCtx$4.default)();
		const e$1 = _usingCtx28.u(exp2(x2));
		return multiply(x1, e$1);
	} catch (_) {
		_usingCtx28.e = _;
	} finally {
		_usingCtx28.d();
	}
}
/**
* Decompose floating-point values into mantissa and two's exponent.
*
* The mantissa is returned in the range `(-1, 1)` with magnitude `>= 0.5` if
* `x != 0`, and the exponent is an integer such that
* `x = mantissa * 2**exponent`.
*/
function frexp(x) {
	try {
		var _usingCtx29 = (0, import_usingCtx$4.default)();
		x = fudgeArray(x);
		const absx = _usingCtx29.u(absolute(x));
		const log2abs = _usingCtx29.u(log2(absx));
		const floorLog = _usingCtx29.u(floor(log2abs));
		const shiftedExp = _usingCtx29.u(floorLog.add(1));
		const exponentI32 = _usingCtx29.u(shiftedExp.astype(require_backend.DType.Int32));
		const exponent = where(equal(x, 0), 0, exponentI32);
		const exponentAsX = _usingCtx29.u(exponent.astype(x.dtype));
		const scale = _usingCtx29.u(exp2(exponentAsX));
		const mantissa = x.div(scale);
		return [mantissa, exponent];
	} catch (_) {
		_usingCtx29.e = _;
	} finally {
		_usingCtx29.d();
	}
}
/** Calculate `2**p` for all p in the input array. */
function exp2(p) {
	try {
		var _usingCtx30 = (0, import_usingCtx$4.default)();
		const prod$2 = _usingCtx30.u(multiply(p, Math.LN2));
		return exp(prod$2);
	} catch (_) {
		_usingCtx30.e = _;
	} finally {
		_usingCtx30.d();
	}
}
/** Return the base-2 logarithm of x, element-wise. */
function log2(x) {
	try {
		var _usingCtx31 = (0, import_usingCtx$4.default)();
		const logX = _usingCtx31.u(log(x));
		return logX.mul(Math.LOG2E);
	} catch (_) {
		_usingCtx31.e = _;
	} finally {
		_usingCtx31.d();
	}
}
/** Return the base-10 logarithm of x, element-wise. */
function log10(x) {
	try {
		var _usingCtx32 = (0, import_usingCtx$4.default)();
		const logX = _usingCtx32.u(log(x));
		return logX.mul(Math.LOG10E);
	} catch (_) {
		_usingCtx32.e = _;
	} finally {
		_usingCtx32.d();
	}
}
/** Calculate `exp(x) - 1` element-wise. */
function expm1(x) {
	try {
		var _usingCtx33 = (0, import_usingCtx$4.default)();
		const expX = _usingCtx33.u(exp(x));
		return expX.sub(1);
	} catch (_) {
		_usingCtx33.e = _;
	} finally {
		_usingCtx33.d();
	}
}
/** Calculate the natural logarithm of `1 + x` element-wise. */
function log1p(x) {
	try {
		var _usingCtx34 = (0, import_usingCtx$4.default)();
		const sum$1 = _usingCtx34.u(add(1, x));
		return log(sum$1);
	} catch (_) {
		_usingCtx34.e = _;
	} finally {
		_usingCtx34.d();
	}
}
/** Convert angles from degrees to radians. */
function deg2rad(x) {
	return multiply(x, pi / 180);
}
/** @function Alias of `jax.numpy.deg2rad()`. */
const radians = deg2rad;
/** Convert angles from radians to degrees. */
function rad2deg(x) {
	return multiply(x, 180 / pi);
}
/** @function Alias of `jax.numpy.rad2deg()`. */
const degrees = rad2deg;
/**
* @function
* Computes first array raised to power of second array, element-wise.
*/
const power = jit$1(function power$1(x1, x2) {
	try {
		var _usingCtx35 = (0, import_usingCtx$4.default)();
		const x2i = trunc(x2);
		const shouldBeNaN = multiply(x2.notEqual(x2i), x1.less(0));
		const parityRaw = _usingCtx35.u(mod(x2i, 2));
		const parityOdd = _usingCtx35.u(parityRaw.notEqual(0));
		const resultSign = where(parityOdd, where(x1.less(0), -1, 1), 1);
		const absX1 = _usingCtx35.u(absolute(x1));
		const logAbsX1 = _usingCtx35.u(log(absX1));
		const scaled = _usingCtx35.u(logAbsX1.mul(x2));
		const magnitude = _usingCtx35.u(exp(scaled));
		const signedMagnitude = _usingCtx35.u(magnitude.mul(resultSign));
		return where(shouldBeNaN, nan, signedMagnitude);
	} catch (_) {
		_usingCtx35.e = _;
	} finally {
		_usingCtx35.d();
	}
});
/** @function Calculate the element-wise cube root of the input array. */
const cbrt = jit$1(function cbrt$1(x) {
	try {
		var _usingCtx36 = (0, import_usingCtx$4.default)();
		const sgn = where(less(x, 0), -1, 1);
		const signedX = _usingCtx36.u(x.mul(sgn));
		const logSignedX = _usingCtx36.u(log(signedX));
		const scaled = _usingCtx36.u(logSignedX.mul(1 / 3));
		const magnitude = _usingCtx36.u(exp(scaled));
		return sgn.mul(magnitude);
	} catch (_) {
		_usingCtx36.e = _;
	} finally {
		_usingCtx36.d();
	}
});
/**
* @function
* Calculate element-wise hyperbolic sine of input.
*
* `sinh(x) = (exp(x) - exp(-x)) / 2`
*/
const sinh = jit$1(function sinh$1(x) {
	try {
		var _usingCtx37 = (0, import_usingCtx$4.default)();
		const ex = exp(x);
		const emx = reciprocal(ex);
		const diff = _usingCtx37.u(ex.sub(emx));
		return diff.mul(.5);
	} catch (_) {
		_usingCtx37.e = _;
	} finally {
		_usingCtx37.d();
	}
});
/**
* @function
* Calculate element-wise hyperbolic cosine of input.
*
* `cosh(x) = (exp(x) + exp(-x)) / 2`
*/
const cosh = jit$1(function cosh$1(x) {
	try {
		var _usingCtx38 = (0, import_usingCtx$4.default)();
		const ex = exp(x);
		const emx = reciprocal(ex);
		const sum$1 = _usingCtx38.u(ex.add(emx));
		return sum$1.mul(.5);
	} catch (_) {
		_usingCtx38.e = _;
	} finally {
		_usingCtx38.d();
	}
});
/**
* @function
* Calculate element-wise hyperbolic tangent of input.
*
* `tanh(x) = sinh(x)/cosh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
*/
const tanh = jit$1(function tanh$1(x) {
	try {
		var _usingCtx39 = (0, import_usingCtx$4.default)();
		const negsgn = where(less(x, 0), 1, -1);
		const signed = _usingCtx39.u(x.mul(negsgn));
		const scaled = _usingCtx39.u(signed.mul(2));
		const en2x = exp(scaled);
		const numer = _usingCtx39.u(en2x.sub(1));
		const denom = _usingCtx39.u(en2x.add(1));
		const ratio = _usingCtx39.u(numer.div(denom));
		return ratio.mul(negsgn);
	} catch (_) {
		_usingCtx39.e = _;
	} finally {
		_usingCtx39.d();
	}
});
/**
* @function
* Calculate element-wise inverse hyperbolic sine of input.
*
* `arcsinh(x) = ln(x + sqrt(x^2 + 1))`
*/
const arcsinh = jit$1(function arcsinh$1(x) {
	try {
		var _usingCtx40 = (0, import_usingCtx$4.default)();
		const sq = _usingCtx40.u(square(x));
		const shifted = _usingCtx40.u(sq.add(1));
		const root = _usingCtx40.u(sqrt(shifted));
		const arg = _usingCtx40.u(x.add(root));
		return log(arg);
	} catch (_) {
		_usingCtx40.e = _;
	} finally {
		_usingCtx40.d();
	}
});
/**
* @function
* Calculate element-wise inverse hyperbolic cosine of input.
*
* `arccosh(x) = ln(x + sqrt(x^2 - 1))`
*/
const arccosh = jit$1(function arccosh$1(x) {
	try {
		var _usingCtx41 = (0, import_usingCtx$4.default)();
		const sq = _usingCtx41.u(square(x));
		const shifted = _usingCtx41.u(sq.sub(1));
		const root = _usingCtx41.u(sqrt(shifted));
		const arg = _usingCtx41.u(x.add(root));
		return log(arg);
	} catch (_) {
		_usingCtx41.e = _;
	} finally {
		_usingCtx41.d();
	}
});
/**
* @function
* Calculate element-wise inverse hyperbolic tangent of input.
*
* `arctanh(x) = 0.5 * ln((1 + x) / (1 - x))`
*/
const arctanh = jit$1(function arctanh$1(x) {
	return log(add(1, x).div(subtract(1, x))).mul(.5);
});
/**
* Compute the variance of an array.
*
* The variance is computed for the flattened array by default, otherwise over
* the specified axis.
*
* If `correction` is provided, the divisor in calculation is `N - correction`,
* where `N` represents the number of elements (e.g., for Bessel's correction).
*/
function var_(x, axis = null, opts) {
	x = fudgeArray(x);
	axis = require_backend.normalizeAxis(axis, x.ndim);
	const n = axis.reduce((acc, a) => acc * x.shape[a], 1);
	if (n === 0) throw new Error("var: cannot compute variance over zero-length axis");
	const d = [];
	try {
		const mu = opts?.mean !== void 0 ? opts.mean : (() => {
			const m = mean(x, axis, { keepdims: true });
			d.push(m);
			return m;
		})();
		const centered = x.sub(mu);
		d.push(centered);
		const sq = square(centered);
		d.push(sq);
		const summed = sq.sum(axis, { keepdims: opts?.keepdims });
		d.push(summed);
		return summed.mul(1 / (n - (opts?.correction ?? 0)));
	} finally {
		for (const v of d) v[Symbol.dispose]();
	}
}
/**
* Compute the standard deviation of an array.
*
* The standard deviation is computed for the flattened array by default,
* otherwise over the specified axis.
*
* If `correction` is provided, the divisor in calculation is `N - correction`,
* where `N` represents the number of elements (e.g., for Bessel's correction).
*/
function std(x, axis = null, opts) {
	try {
		var _usingCtx42 = (0, import_usingCtx$4.default)();
		const v = _usingCtx42.u(var_(x, axis, opts));
		return sqrt(v);
	} catch (_) {
		_usingCtx42.e = _;
	} finally {
		_usingCtx42.d();
	}
}
/** Estimate the sample covariance of a set of variables. */
function cov(x, y = null, { rowvar = true } = {}) {
	try {
		var _usingCtx43 = (0, import_usingCtx$4.default)();
		const disposables = [];
		let a = fudgeArray(x);
		if (a.ndim === 1) {
			a = a.reshape([1, a.shape[0]]);
			disposables.push(a);
		}
		if (y !== null) {
			let b = fudgeArray(y);
			if (b.ndim === 1) {
				b = b.reshape([1, b.shape[0]]);
				disposables.push(b);
			}
			a = vstack([a, b]);
			disposables.push(a);
		}
		if (!rowvar) {
			a = a.transpose();
			disposables.push(a);
		}
		const [_M, N] = a.shape;
		const mean$1 = _usingCtx43.u(a.mean(1, { keepdims: true }));
		const centered = a.sub(mean$1);
		disposables.push(centered);
		const xt = _usingCtx43.u(centered.transpose());
		const dotResult = _usingCtx43.u(dot$1(centered, xt));
		const result = dotResult.div(N - 1);
		for (const d of disposables) d.dispose();
		return result;
	} catch (_) {
		_usingCtx43.e = _;
	} finally {
		_usingCtx43.d();
	}
}
/** Compute the Pearson correlation coefficients (in range `[-1, 1]`). */
function corrcoef(x, y) {
	try {
		var _usingCtx44 = (0, import_usingCtx$4.default)();
		const c = _usingCtx44.u(cov(x, y));
		const variances = _usingCtx44.u(diag(c));
		const norm = _usingCtx44.u(sqrt(outer(variances, variances)));
		return c.div(norm);
	} catch (_) {
		_usingCtx44.e = _;
	} finally {
		_usingCtx44.d();
	}
}
/** Test element-wise for positive or negative infinity, return bool array. */
function isinf(x) {
	try {
		var _usingCtx45 = (0, import_usingCtx$4.default)();
		x = fudgeArray(x);
		if (!require_backend.isFloatDtype(x.dtype)) return fullLike$1(x, false);
		const posInf = _usingCtx45.u(x.equal(Infinity));
		const negInf = _usingCtx45.u(x.equal(-Infinity));
		return posInf.add(negInf);
	} catch (_) {
		_usingCtx45.e = _;
	} finally {
		_usingCtx45.d();
	}
}
/** Test element-wise for NaN (Not a Number). */
function isnan(x) {
	x = fudgeArray(x);
	return require_backend.isFloatDtype(x.dtype) ? x.notEqual(x) : fullLike$1(x, false);
}
/** Test element-wise for negative infinity, return bool array. */
function isneginf(x) {
	x = fudgeArray(x);
	return require_backend.isFloatDtype(x.dtype) ? x.equal(-Infinity) : fullLike$1(x, false);
}
/** Test element-wise for positive infinity, return bool array. */
function isposinf(x) {
	x = fudgeArray(x);
	return require_backend.isFloatDtype(x.dtype) ? x.equal(Infinity) : fullLike$1(x, false);
}
/**
* Replace NaN and infinite entries in an array.
*
* By default, NaNs are replaced with `0.0`, and infinities are are substituted
* with the corresponding maximum or minimum finite values.
*/
function nanToNum(x, { nan: nan$1 = 0, posinf = null, neginf = null } = {}) {
	try {
		var _usingCtx46 = (0, import_usingCtx$4.default)();
		x = fudgeArray(x);
		const nanMask = _usingCtx46.u(isnan(x));
		const afterNan = _usingCtx46.u(where(nanMask, nan$1, x));
		posinf ??= require_backend.isFloatDtype(afterNan.dtype) ? finfo(afterNan.dtype).max : iinfo(afterNan.dtype).max;
		neginf ??= require_backend.isFloatDtype(afterNan.dtype) ? finfo(afterNan.dtype).min : iinfo(afterNan.dtype).min;
		const posInfMask = _usingCtx46.u(isposinf(afterNan));
		const afterPosInf = _usingCtx46.u(where(posInfMask, posinf, afterNan));
		const negInfMask = _usingCtx46.u(isneginf(afterPosInf));
		return where(negInfMask, neginf, afterPosInf);
	} catch (_) {
		_usingCtx46.e = _;
	} finally {
		_usingCtx46.d();
	}
}
/**
* @function
* Test element-wise for finite values (not infinity or NaN).
*/
const isfinite = jit$1(function isfinite$1(x) {
	try {
		var _usingCtx47 = (0, import_usingCtx$4.default)();
		if (!require_backend.isFloatDtype(x.dtype)) return fullLike$1(x, true);
		const nanMask = _usingCtx47.u(isnan(x));
		const infMask = _usingCtx47.u(isinf(x));
		const union = _usingCtx47.u(nanMask.add(infMask));
		return union.notEqual(true);
	} catch (_) {
		_usingCtx47.e = _;
	} finally {
		_usingCtx47.d();
	}
});

//#endregion
//#region src/library/lax-linalg.ts
var lax_linalg_exports = {};
__export(lax_linalg_exports, {
	cholesky: () => cholesky$1,
	lu: () => lu,
	triangularSolve: () => triangularSolve
});
var import_usingCtx$3 = /* @__PURE__ */ __toESM(require_usingCtx(), 1);
/**
* Compute the Cholesky decomposition of a symmetric positive-definite matrix.
*
* The Cholesky decomposition of a matrix `A` is:
*
* - A = L @ L^T  (for upper=false, default)
* - A = U^T @ U  (for upper=true)
*
* where `L` is a lower-triangular matrix and `U` is an upper-triangular matrix.
* The input matrix must be symmetric and positive-definite.
*
* @example
* ```ts
* import { lax, numpy as np } from "@jax-js/jax";
*
* const x = np.array([[2., 1.], [1., 2.]]);
*
* // Lower Cholesky factorization (default):
* const L = lax.linalg.cholesky(x);
* // L ≈ [[1.4142135, 0], [0.70710677, 1.2247449]]
*
* // Upper Cholesky factorization:
* const U = lax.linalg.cholesky(x, { upper: true });
* // U ≈ [[1.4142135, 0.70710677], [0, 1.2247449]]
* ```
*/
function cholesky$1(a, { upper = false } = {}) {
	if (upper) try {
		var _usingCtx$1 = (0, import_usingCtx$3.default)();
		const L = _usingCtx$1.u(cholesky$2(a));
		return moveaxis$1(L, -2, -1);
	} catch (_) {
		_usingCtx$1.e = _;
	} finally {
		_usingCtx$1.d();
	}
	return cholesky$2(a);
}
/**
* LU decomposition with partial pivoting.
*
* Computes the matrix decomposition: `P @ A = L @ U`, where `P` is a
* permutation of the rows of `A`, `L` is lower-triangular with unit diagonal,
* and `U` is upper-triangular.
*
* @param x - A batch of matrices with shape `[..., m, n]`.
*
* @returns A tuple `(lu, pivots, permutation)` where:
* - `lu`: combined lower and upper triangular matrices.
* - `pivots`: an array of pivot indices with shape `[..., min(m, n)]`.
* - `permutation`: the permutation generated by pivots with shape `[..., m]`.
*
* @example
* ```ts
* import { lax, numpy as np } from "@jax-js/jax";
*
* const A = np.array([[4., 3.], [6., 3.]]);
* const [lu, pivots, permutation] = lax.linalg.lu(A);
* // lu ≈ [[6., 3.], [0.6666667, 1.0]]
* // pivots = [1, 1]
* // permutation = [1, 0]
* ```
*/
function lu(x) {
	return lu$1(x);
}
/**
* Solve a triangular linear system.
*
* Solves `a @ x = b` (if leftSide=true) or `x @ a = b` (if leftSide=false)
* where `a` is a triangular matrix.
*
* @example
* ```ts
* import { lax, numpy as np } from "@jax-js/jax";
*
* const L = np.array([[2., 0.], [1., 3.]]);
* const b = np.array([4., 7.]).reshape([2, 1]);
*
* // Solve L @ x = b
* const x = lax.linalg.triangularSolve(L, b, { leftSide: true, lower: true });
* // x = [[2.], [5./3.]]
* ```
*/
function triangularSolve(a, b, { leftSide = false, lower = false, transposeA = false, unitDiagonal = false } = {}) {
	a = fudgeArray(a);
	b = fudgeArray(b);
	const d = [];
	try {
		if (!leftSide) transposeA = !transposeA;
		else {
			b = moveaxis$1(b, -2, -1);
			d.push(b);
		}
		if (transposeA) {
			a = moveaxis$1(a, -2, -1);
			d.push(a);
			lower = !lower;
		}
		let x = triangularSolve$1(a, b, {
			lower,
			unitDiagonal
		});
		if (leftSide) {
			d.push(x);
			x = moveaxis$1(x, -2, -1);
		}
		return x;
	} finally {
		for (const v of d) v[Symbol.dispose]();
	}
}

//#endregion
//#region src/library/lax-scan.ts
/**
* Scan a function over leading array axes while carrying along state.
*
* Think of `scan` as a functional `reduce` that also returns all intermediate
* results. It iterates over the leading axis of `xs`, threading a "carry" value
* through each step and collecting outputs.
*
* ## Type Signature
*
* ```ts
* scan(f, init, xs) → [finalCarry, ys]
* scan(f, init, null, { length }) → [finalCarry, ys]  // carry-only scan
*
* // Where:
* // f: (carry: C, x: X | null) => [C, Y | null]  -- step function
* // init: C                               -- initial carry
* // xs: X[] | null                        -- input array or null for carry-only
* // finalCarry: C                         -- carry after last iteration
* // ys: Y[] | null                        -- stacked outputs (null if Y=null)
* ```
*
* ## Semantics
*
* The semantics are roughly equivalent to this JavaScript:
* ```ts
* function scan(f, init, xs) {
*   let carry = init;
*   const ys = [];
*   for (const x of xs) {
*     const [newCarry, y] = f(carry, x);
*     carry = newCarry;
*     ys.push(y);
*   }
*   return [carry, np.stack(ys)];
* }
* ```
*
* Unlike a plain JavaScript loop:
* - Both `xs` and `ys` can be arbitrary pytrees (nested objects/arrays)
* - The scan is compiled to efficient native code (WASM/WebGPU)
* - Supports autodiff: `grad(f)` works through scan
* - The carry shape/dtype must be fixed across all iterations
*
* ## Ownership
*
* **Inputs:** `init` and `xs` are NOT consumed. Use `using` or `.dispose()` to
* manage their lifetime if needed.
*
* **Body function:**
* - `carry` and `x` are **managed** by scan — do NOT manually dispose them
* - Operations do NOT consume inputs, so no `.ref` is needed inside the body
*
* **Example — simple body:**
* ```ts
* const step = (carry, x) => {
*   const newCarry = np.add(carry, x);
*   return [newCarry, carry];  // carry used freely in multiple places
* };
* ```
*
* **Outputs (caller owns):**
* - `finalCarry` and `ys` are owned by caller — dispose when done
*
* @param f - Step function `(carry, x) => [newCarry, y]` where:
*   - `carry` is the current state (same structure as `init`)
*   - `x` is a slice of `xs` along axis 0, or `null` if `xs` is null
*   - `newCarry` is the updated state (same structure/shape as `carry`)
*   - `y` is the output for this iteration, or `null` to skip output stacking
* @param init - Initial carry value. Can be a single array or a pytree of arrays.
* @param xs - Input sequence to scan over, or `null` for carry-only scans.
*   When an array/pytree, the leading axis is the scan dimension.
*   When `null`, you must provide `{ length }` in options.
* @param options - Scan options
* @returns `[finalCarry, ys]` where:
*   - `finalCarry` has the same structure as `init`
*   - `ys` has the same structure as `y` from `f`, with each leaf having
*     an additional leading axis of size `length`. If `y` is `null`, `ys` is `null`
*     (no memory allocated for outputs).
*
* @example Cumulative sum
* ```ts
* import { lax, numpy as np } from '@jax-js/jax';
*
* const step = (carry, x) => {
*   const sum = np.add(carry, x);
*   return [sum, sum];  // same array can be used in both outputs
* };
*
* const init = np.array([0.0]);
* const xs = np.array([[1], [2], [3], [4], [5]]);
* const [final, sums] = await lax.scan(step, init, xs);
*
* console.log(await final.data());  // [15]
* console.log(await sums.data());   // [[1], [3], [6], [10], [15]]
*
* final.dispose();
* sums.dispose();
* ```
*
* @example Factorial via scan
* ```ts
* // Compute n! for n = 1..5
* const step = (carry, x) => {
*   const next = np.multiply(carry, x);
*   return [next, next];
* };
*
* const init = np.array([1]);
* const xs = np.array([[1], [2], [3], [4], [5]]);
* const [final, factorials] = await lax.scan(step, init, xs);
* // factorials = [[1], [2], [6], [24], [120]]
* ```
*
* @example Pytree carry (multiple state variables)
* ```ts
* // Track both sum and count
* const step = (carry, x) => {
*   const newSum = np.add(carry.sum, x);
*   const newCount = np.add(carry.count, np.array([1]));
*   return [
*     { sum: newSum, count: newCount },
*     { sum: newSum, count: newCount }
*   ];
* };
*
* const init = { sum: np.array([0]), count: np.array([0]) };
* const xs = np.array([[10], [20], [30]]);
* const [final, history] = await lax.scan(step, init, xs);
* // final.sum = [60], final.count = [3]
* ```
*
* @example Reverse scan
* ```ts
* // Process sequence from end to beginning
* const [final, ys] = await lax.scan(step, init, xs, { reverse: true });
* ```
*
* ## jax-js Extensions
*
* These features extend JAX's scan API for TypeScript/JavaScript ergonomics:
*
* ### xs=null (carry-only scan)
*
* Pass `null` as `xs` with `{ length }` to iterate without input arrays.
* Useful for generators, RNG sequences, Fibonacci, or any state-only iteration.
* The body receives `null` as the second argument.
*
* ### Y=null (skip output stacking)
*
* Return `[newCarry, null]` from the body to skip allocating stacked outputs.
* Useful when you only need the final carry (e.g., Mandelbrot iteration counts).
* The returned `ys` will be `null`, saving memory for large iteration counts.
*
* @example xs=null: Carry-only scan
* ```ts
* // Generate a sequence without allocating input arrays
* const step = (carry, _x) => {
*   const next = np.add(carry, np.array([1.0]));
*   return [next, carry];  // output is old carry value
* };
*
* const init = np.array([0.0]);
* const [final, ys] = await lax.scan(step, init, null, { length: 5 });
* // ys = [[0], [1], [2], [3], [4]], final = [5]
* ```
*
* @example Y=null: Skip output stacking
* ```ts
* // Only need final carry, not intermediate outputs (saves memory)
* const step = (carry, x) => {
*   const Asq = carry.A.mul(carry.A);
*   const newA = Asq.add(x);
*   const newCount = carry.count.add(Asq.less(100).astype(np.int32));
*   return [{ A: newA, count: newCount }, null];  // null skips Y stacking
* };
*
* const init = { A: np.zeros([100]), count: np.zeros([100], np.int32) };
* const [final, ys] = await lax.scan(step, init, xs);
* // ys is null — no memory allocated for intermediate outputs
* ```
*
* @example jit(scan) - Compile the entire scan loop
* ```ts
* import { jit, lax, numpy as np } from '@jax-js/jax';
*
* // Wrap scan in jit to compile the entire loop into optimized native code.
* // This is the most common and efficient pattern for production use.
* const step = (carry, x) => {
*   const newCarry = np.add(carry, x);
*   return [newCarry, newCarry];
* };
*
* const scanFn = jit((init, xs) => lax.scan(step, init, xs));
*
* const init = np.array([0.0]);
* const xs = np.array([[1.0], [2.0], [3.0]]);
* const [final, ys] = await scanFn(init, xs);
*
* console.log(await final.data());  // [6]
* scanFn.dispose();  // Free compiled program
* ```
*
* @example scan(jit(body)) - JIT-compile only the step function
* ```ts
* import { jit, lax, numpy as np } from '@jax-js/jax';
*
* // JIT-compile just the step function. Each iteration calls compiled code,
* // but the loop itself runs in JavaScript. Useful when step is expensive
* // but you want to inspect intermediate values or the scan body is dynamic.
* const step = jit((carry, x) => {
*   const newCarry = np.add(carry, x);
*   return [newCarry, newCarry];
* });
*
* const init = np.array([0.0]);
* const xs = np.array([[1.0], [2.0], [3.0]]);
* const [final, ys] = await lax.scan(step, init, xs);
*
* console.log(await final.data());  // [6]
* step.dispose();  // Free compiled step function
* ```
*
* @example With grad for differentiation
* ```ts
* import { grad, lax, numpy as np } from '@jax-js/jax';
*
* const loss = (init, xs) => {
*   const [final, ys] = lax.scan(step, init, xs);
*   final.dispose();
*   return np.sum(ys);
* };
*
* const gradLoss = grad(loss);
* const [dInit, dXs] = await gradLoss(init, xs);
* ```
*
* @see {@link https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html | JAX lax.scan}
*/
function scan(f, init$1, xs, options) {
	const opts = options ?? {};
	const { length: lengthOpt, reverse = false, acceptPath, checkpoint } = opts;
	const xsIsNull = xs === null;
	const [initFlat, initTreedef] = flatten(init$1);
	const [xsFlat, xsTreedef] = xsIsNull ? [[], null] : flatten(xs);
	const n = lengthOpt ?? (xsFlat.length > 0 ? xsFlat[0].shape[0] : 0);
	const carryAvals = initFlat.map((arr) => ShapedArray.fromAval(getAval(arr)));
	const xSliceAvals = xsFlat.map((arr) => {
		const aval = getAval(arr);
		return new ShapedArray(aval.shape.slice(1), aval.dtype, aval.weakType);
	});
	let yTreedef_;
	const flatF = (carryFlat, xSliceFlat) => {
		const carry = unflatten(initTreedef, carryFlat);
		const xSlice = xsIsNull ? xs : unflatten(xsTreedef, xSliceFlat);
		const [newCarry, y] = f(carry, xSlice);
		const [newCarryFlat] = flatten(newCarry);
		const [yFlat, yTreedef] = flatten(y);
		yTreedef_ = yTreedef;
		return [newCarryFlat, yFlat];
	};
	const traceFn = (...args) => {
		const numCarry$1 = carryAvals.length;
		const carryFlat = args.slice(0, numCarry$1);
		const xSliceFlat = args.slice(numCarry$1);
		const [newCarryFlat, yFlat] = flatF(carryFlat, xSliceFlat);
		return [...newCarryFlat, ...yFlat];
	};
	const traceAvals = [...carryAvals, ...xSliceAvals];
	setScanBodyTraceActive(true);
	let closedJaxpr, _outTreedef;
	try {
		({jaxpr: closedJaxpr, treedef: _outTreedef} = makeJaxpr$1(traceFn)(...traceAvals));
	} finally {
		setScanBodyTraceActive(false);
	}
	const jaxpr = closedJaxpr.jaxpr;
	const consts = closedJaxpr.consts;
	if (n === 0) {
		if (xsIsNull && lengthOpt === void 0) throw new Error("scan: length option is required when xs is null");
		closedJaxpr.dispose();
		const finalCarryFlat = initFlat;
		const numCarry$1 = initFlat.length;
		const yOutAtoms = jaxpr.outs.slice(numCarry$1);
		const yFlatEmpty = yOutAtoms.map((atom) => {
			const aval = atom.aval;
			return zeros([0, ...aval.shape], { dtype: aval.dtype });
		});
		const finalCarry$1 = unflatten(initTreedef, finalCarryFlat);
		const ys$1 = yTreedef_ === JsTreeDef.none ? null : unflatten(yTreedef_, yFlatEmpty);
		return [finalCarry$1, ys$1];
	}
	const scanArgs = [
		...consts,
		...initFlat,
		...xsFlat
	];
	const numCarry = initFlat.length;
	const numConsts = consts.length;
	const results = bind(Primitive.Scan, scanArgs, {
		jaxpr,
		numCarry,
		numConsts,
		length: n,
		reverse,
		acceptPath,
		checkpoint
	});
	const carryOut = results.slice(0, numCarry);
	const ysFlat = results.slice(numCarry);
	closedJaxpr.dispose();
	const finalCarry = unflatten(initTreedef, carryOut);
	const ys = unflatten(yTreedef_, ysFlat);
	return [finalCarry, ys];
}

//#endregion
//#region src/library/lax.ts
var lax_exports = {};
__export(lax_exports, {
	conv: () => conv,
	convGeneralDilated: () => convGeneralDilated,
	convTranspose: () => convTranspose,
	convWithGeneralPadding: () => convWithGeneralPadding,
	dot: () => dot,
	erf: () => erf,
	erfc: () => erfc,
	linalg: () => lax_linalg_exports,
	reduceWindow: () => reduceWindow,
	scan: () => scan,
	stopGradient: () => stopGradient$1,
	topK: () => topK
});
var import_usingCtx$2 = /* @__PURE__ */ __toESM(require_usingCtx(), 1);
const JsArray = globalThis.Array;
/**
* General dot product/contraction operator.
*
* Prefer higher-level functions like `jax.numpy.dot()`, `jax.numpy.matmul()`,
* `jax.numpy.tensordot(), and `jax.numpy.einsum()` where possible.
*/
function dot(lhs, rhs, { lhsContractingDims: lc = [], rhsContractingDims: rc = [], lhsBatchDims: lb = [], rhsBatchDims: rb = [] } = {}) {
	try {
		var _usingCtx$1 = (0, import_usingCtx$2.default)();
		if (lc.length !== rc.length) throw new Error(`dot: contracting dims lengths mismatch, got ${JSON.stringify(lc)} and ${JSON.stringify(rc)}`);
		else if (lb.length !== rb.length) throw new Error(`dot: batch dims lengths mismatch, got ${JSON.stringify(lb)} and ${JSON.stringify(rb)}`);
		lc = lc.map((a) => require_backend.checkAxis(a, lhs.ndim));
		rc = rc.map((a) => require_backend.checkAxis(a, rhs.ndim));
		lb = lb.map((a) => require_backend.checkAxis(a, lhs.ndim));
		rb = rb.map((a) => require_backend.checkAxis(a, rhs.ndim));
		if (lc.some((a) => lb.includes(a))) throw new Error(`dot: lhs contracting dims ${JSON.stringify(lc)} overlap with batch dims ${JSON.stringify(lb)}`);
		else if (rc.some((a) => rb.includes(a))) throw new Error(`dot: rhs contracting dims ${JSON.stringify(rc)} overlap with batch dims ${JSON.stringify(rb)}`);
		const lf = require_backend.range(lhs.ndim).filter((a) => !lc.includes(a) && !lb.includes(a));
		const rf = require_backend.range(rhs.ndim).filter((a) => !rc.includes(a) && !rb.includes(a));
		const lhs2 = _usingCtx$1.u(lhs.transpose([
			...lb,
			...lf,
			...lc
		]));
		const rhs2 = _usingCtx$1.u(rhs.transpose([
			...rb,
			...rf,
			...rc
		]));
		if (lc.length === 0) try {
			var _usingCtx3 = (0, import_usingCtx$2.default)();
			const lhsR$1 = _usingCtx3.u(lhs2.reshape([
				...lb.map((a) => lhs.shape[a]),
				...lf.map((a) => lhs.shape[a]),
				...require_backend.rep(rf.length, 1)
			]));
			const rhsR$1 = _usingCtx3.u(rhs2.reshape([
				...rb.map((a) => rhs.shape[a]),
				...require_backend.rep(lf.length, 1),
				...rf.map((a) => rhs.shape[a])
			]));
			return mul(lhsR$1, rhsR$1);
		} catch (_) {
			_usingCtx3.e = _;
		} finally {
			_usingCtx3.d();
		}
		const dotShapeX = lc.map((a) => lhs.shape[a]);
		const dotShapeY = rc.map((a) => rhs.shape[a]);
		if (!require_backend.deepEqual(dotShapeX, dotShapeY)) throw new Error(`dot: shapes not aligned along contracting dims: ${JSON.stringify(dotShapeX)} != ${JSON.stringify(dotShapeY)}`);
		const lhsR = _usingCtx$1.u(lhs2.reshape([
			...lb.map((a) => lhs.shape[a]),
			...lf.map((a) => lhs.shape[a]),
			...require_backend.rep(rf.length, 1),
			require_backend.prod(dotShapeX)
		]));
		const rhsR = _usingCtx$1.u(rhs2.reshape([
			...rb.map((a) => rhs.shape[a]),
			...require_backend.rep(lf.length, 1),
			...rf.map((a) => rhs.shape[a]),
			require_backend.prod(dotShapeY)
		]));
		return dot$2(lhsR, rhsR);
	} catch (_) {
		_usingCtx$1.e = _;
	} finally {
		_usingCtx$1.d();
	}
}
function padtypeToPads(inShape, filterShape, strides, dilation, padding) {
	const padType = padding.toUpperCase();
	switch (padType) {
		case "VALID": return require_backend.rep(inShape.length, [0, 0]);
		case "SAME":
		case "SAME_LOWER": {
			const outShape = inShape.map((size$1, i) => Math.ceil(size$1 / strides[i]));
			const padSizes = require_backend.zipn(outShape, strides, filterShape, dilation, inShape).map(([o, s, k, d, i]) => Math.max(0, (o - 1) * s + 1 + (k - 1) * d - i));
			if (padType === "SAME") return padSizes.map((size$1) => [size$1 >> 1, size$1 - (size$1 >> 1)]);
			else return padSizes.map((size$1) => [size$1 - (size$1 >> 1), size$1 >> 1]);
		}
		default: throw new Error(`Unknown padding type: ${padType}`);
	}
}
/**
* General n-dimensional convolution operator, with optional dilation.
*
* The semantics of this operation mimic the `jax.lax.conv_general_dilated`
* function in JAX, which wraps XLA's general convolution operator.
*
* @param lhs - Input tensor; shape `[N, C_in, ...xs]`
* @param rhs - Convolution kernel; shape `[C_out, C_in / G, ...ks]`
* @param windowStrides - Strides for each spatial dimension
* @param padding - Padding for each spatial dimension, or a string
*   (`"VALID"`, `"SAME"`, or `"SAME_LOWER"`)
*/
function convGeneralDilated(lhs, rhs, windowStrides, padding, { lhsDilation, rhsDilation, featureGroupCount = 1 } = {}) {
	if (lhs.ndim < 2) throw new Error("lhs must have at least 2 dimensions");
	if (rhs.ndim < 2) throw new Error("rhs must have at least 2 dimensions");
	if (typeof padding === "string") {
		if (lhsDilation?.some((d) => d !== 1)) throw new Error("String padding is not supported for transposed convolutions");
		padding = padtypeToPads(lhs.shape.slice(2), rhs.shape.slice(2), windowStrides, rhsDilation ?? require_backend.rep(rhs.ndim - 2, 1), padding);
	}
	if (featureGroupCount !== 1) try {
		var _usingCtx4 = (0, import_usingCtx$2.default)();
		const G = featureGroupCount;
		const [N, C_in, ...xs] = lhs.shape;
		const [C_out, C_in_per_group, ...ks] = rhs.shape;
		if (C_in % G !== 0) throw new Error(`featureGroupCount=${G} must divide input channels=${C_in}`);
		if (C_out % G !== 0) throw new Error(`featureGroupCount=${G} must divide output channels=${C_out}`);
		if (C_in / G !== C_in_per_group) throw new Error(`rhs input channels=${C_in_per_group} must equal lhs input channels / groups=${C_in / G}`);
		const lhsReshaped = _usingCtx4.u(lhs.reshape([
			N,
			G,
			C_in / G,
			...xs
		]));
		const lhsGrouped = _usingCtx4.u(moveaxis(lhsReshaped, 1, 0));
		const rhsGrouped = _usingCtx4.u(rhs.reshape([
			G,
			C_out / G,
			C_in_per_group,
			...ks
		]));
		const result = _usingCtx4.u(conv$1(lhsGrouped, rhsGrouped, {
			vmapDims: 1,
			strides: windowStrides,
			padding,
			lhsDilation,
			rhsDilation
		}));
		const ys = result.shape.slice(3);
		const moved = _usingCtx4.u(moveaxis(result, 0, 1));
		return moved.reshape([
			N,
			C_out,
			...ys
		]);
	} catch (_) {
		_usingCtx4.e = _;
	} finally {
		_usingCtx4.d();
	}
	return conv$1(lhs, rhs, {
		strides: windowStrides,
		padding,
		lhsDilation,
		rhsDilation
	});
}
/** Convenience wrapper around `convGeneralDilated`. */
function convWithGeneralPadding(lhs, rhs, windowStrides, padding, lhsDilation, rhsDilation) {
	return convGeneralDilated(lhs, rhs, windowStrides, padding, {
		lhsDilation,
		rhsDilation
	});
}
/** Convenience wrapper around `convGeneralDilated`. */
function conv(lhs, rhs, windowStrides, padding) {
	return convGeneralDilated(lhs, rhs, windowStrides, padding);
}
/**
* Convenience wrapper for calculating the N-d convolution "transpose".
*
* This function directly calculates a fractionally strided conv rather than
* indirectly calculating the gradient (transpose) of a forward convolution.
* It is equivalent to the JAX version, except:
*
* - The `use_consistent_padding` option is not available. We only have the
*   consistent padding case (JAX version >0.8.4).
* - The order of dimensions matches `lax.conv_general_dilated`.
*
* Unlike PyTorch/TensorFlow, by default we don't reverse the kernel's spatial
* dimensions or the `(C_out, C_in)` axis order. To get this behavior, set
* `transposeKernel` to true.
*
* @param lhs - Input tensor; shape `[N, C_in, ...xs]`
* @param rhs - Convolution kernel; shape `[C_out, C_in, ...ks]`
* @param strides - Sequence of n integers, sets fractional stride
* @param padding - Apply padding of `dilation * (kernel_size - 1) - padding` to
*   each side of the input, so it acts like gradient of `conv()`
* @param rhsDilation - Atrous dilation for the kernel
* @param transposeKernel - Flip spatial axes and swap the input/output channels
*   of the kernel; its shape should be `[C_in, C_out, ...ks]`
*/
function convTranspose(lhs, rhs, strides, padding, { rhsDilation, transposeKernel = false } = {}) {
	const kernelShape = rhs.shape.slice(2);
	rhsDilation = rhsDilation ?? require_backend.rep(kernelShape.length, 1);
	const effectiveKernel = kernelShape.map((k, i) => Math.max(0, (k - 1) * rhsDilation[i] + 1));
	const pads = effectiveKernel.map((k, i) => convTransposePadding(k, strides[i], typeof padding === "string" ? padding : padding[i]));
	if (transposeKernel) try {
		var _usingCtx5 = (0, import_usingCtx$2.default)();
		const flipped = _usingCtx5.u(flip$1(rhs, require_backend.range(2, rhs.ndim)));
		rhs = moveaxis(flipped, 0, 1);
	} catch (_) {
		_usingCtx5.e = _;
	} finally {
		_usingCtx5.d();
	}
	return convGeneralDilated(lhs, rhs, require_backend.rep(lhs.ndim - 2, 1), pads, {
		lhsDilation: strides,
		rhsDilation
	});
}
function convTransposePadding(k, s, padding) {
	let padLen;
	let pad1;
	if (padding === "SAME") {
		padLen = k + s - 2;
		pad1 = s > k - 1 ? k - 1 : Math.ceil(padLen / 2);
	} else if (padding === "VALID") {
		padLen = k + s - 2 + Math.max(k - s, 0);
		pad1 = k - 1;
	} else if (JsArray.isArray(padding)) {
		const pads = [k - 1 - padding[0], k - 1 - padding[1]];
		pad1 = pads[0];
		padLen = pads[0] + pads[1];
	} else throw new Error(`convTranspose: Invalid padding type ${padding}`);
	return [pad1, padLen - pad1];
}
/** Reduce a computation over padded windows. */
function reduceWindow(operand, computation, windowDimensions, windowStrides) {
	if (operand.ndim < windowDimensions.length) throw new Error(`Operand dimensions ${operand.ndim} < window ${windowDimensions.length}`);
	if (!windowStrides) windowStrides = require_backend.rep(windowDimensions.length, 1);
	for (let i = 0; i < operand.ndim; i++) computation = vmap$1(computation, 0);
	const pooled = bind1(Primitive.Pool, [operand], {
		window: windowDimensions,
		strides: windowStrides
	});
	const result = computation(pooled);
	if (pooled !== result) pooled[Symbol.dispose]?.();
	return result;
}
/** The error function: `erf(x) = 2/sqrt(pi) * int[0..x] exp(-t^2) dt`. */
function erf(x) {
	return erf$1(x);
}
/**
* The complementary error function: `erfc(x) = 1 - erf(x)`.
*
* This function is more accurate than `1 - erf(x)` for large values of `x`,
* where `erf(x)` is very close to 1.
*/
function erfc(x) {
	return erfc$1(x);
}
/**
* Stops gradient computation.
*
* Behaves as the identity function but prevents the flow of gradients during
* forward or reverse-mode automatic differentiation.
*/
function stopGradient$1(x) {
	return stopGradient(x);
}
/**
* Returns top `k` values and their indices along the specified axis of operand.
*
* This is a _stable_ algorithm: If two elements are equal, the lower-index
* element appears first.
*
* @returns A tuple of `(values, indices)`, where `values` and `indices` have
* the same shape as `x`, except along `axis` where they have size `k`.
*/
function topK(x, k, axis = -1) {
	x = fudgeArray(x);
	axis = require_backend.checkAxis(axis, x.ndim);
	const size$1 = x.shape[axis];
	if (k < 0 || k > size$1) throw new Error(`topK: k must be in the range [0, ${size$1}], got ${k}`);
	if (k === 0) {
		const outShape = x.shape.slice();
		outShape[axis] = 0;
		const y$1 = zerosLike$1(x, { shape: outShape });
		const yi$1 = zerosLike$1(x, {
			dtype: require_backend.DType.Int32,
			shape: outShape
		});
		return [y$1, yi$1];
	}
	const flipped = flip$1(x, [axis]);
	const moved = moveaxis(flipped, axis, -1);
	const argsortResult = argsort$1(moved);
	const y = argsortResult[0];
	const yi = argsortResult[1];
	const extract = (a) => {
		const sliced = a.slice(...require_backend.rep(a.ndim - 1, []), [-k]);
		const movedBack = moveaxis(sliced, -1, axis);
		const result$1 = flip$1(movedBack, [axis]);
		if (movedBack !== sliced) movedBack.dispose();
		sliced.dispose();
		return result$1;
	};
	const neg$1 = yi.neg();
	const adjusted = neg$1.add(size$1 - 1);
	const result = [extract(y), extract(adjusted)];
	adjusted.dispose();
	neg$1.dispose();
	yi.dispose();
	y.dispose();
	if (moved !== flipped) moved.dispose();
	flipped.dispose();
	return result;
}

//#endregion
//#region src/library/nn.ts
var nn_exports = {};
__export(nn_exports, {
	celu: () => celu,
	dotProductAttention: () => dotProductAttention,
	elu: () => elu,
	gelu: () => gelu,
	glu: () => glu,
	hardSigmoid: () => hardSigmoid,
	hardSilu: () => hardSilu,
	hardSwish: () => hardSilu,
	hardTanh: () => hardTanh,
	identity: () => identity,
	leakyRelu: () => leakyRelu,
	logSigmoid: () => logSigmoid,
	logSoftmax: () => logSoftmax,
	logmeanexp: () => logmeanexp,
	logsumexp: () => logsumexp,
	mish: () => mish,
	oneHot: () => oneHot,
	relu: () => relu,
	relu6: () => relu6,
	selu: () => selu,
	sigmoid: () => sigmoid,
	silu: () => silu,
	softSign: () => softSign,
	softmax: () => softmax,
	softplus: () => softplus,
	sparsePlus: () => sparsePlus,
	sparseSigmoid: () => sparseSigmoid,
	squareplus: () => squareplus,
	standardize: () => standardize,
	swish: () => silu
});
var import_usingCtx$1 = /* @__PURE__ */ __toESM(require_usingCtx(), 1);
/**
* Rectified Linear Unit (ReLU) activation function:
* `relu(x) = max(x, 0)`.
*/
function relu(x) {
	return maximum(x, 0);
}
/**
* Rectified Linear Unit 6 (ReLU6) activation function:
* `relu6(x) = min(max(x, 0), 6)`.
*/
function relu6(x) {
	return clip(x, 0, 6);
}
/**
* Sigmoid activation function, computed element-wise:
* `sigmoid(x) = 1 / (1 + exp(-x))`.
*
* Reference: https://en.wikipedia.org/wiki/Sigmoid_function
*/
function sigmoid(x) {
	try {
		var _usingCtx$1 = (0, import_usingCtx$1.default)();
		const neg$1 = _usingCtx$1.u(negative(x));
		const e$1 = _usingCtx$1.u(exp(neg$1));
		const sum$1 = _usingCtx$1.u(e$1.add(1));
		return reciprocal(sum$1);
	} catch (_) {
		_usingCtx$1.e = _;
	} finally {
		_usingCtx$1.d();
	}
}
/**
* Softplus activation function:
* `softplus(x) = log(1 + exp(x))`.
*
* Reference: https://en.wikipedia.org/wiki/Softplus
*/
function softplus(x) {
	try {
		var _usingCtx3 = (0, import_usingCtx$1.default)();
		const e$1 = _usingCtx3.u(exp(x));
		const sum$1 = _usingCtx3.u(e$1.add(1));
		return log(sum$1);
	} catch (_) {
		_usingCtx3.e = _;
	} finally {
		_usingCtx3.d();
	}
}
/**
* @function
* Sparse plus function:
*
* - When `x <= -1`: `0`
* - When `-1 < x < 1`: `(x+1)**2 / 4`
* - When `x >= 1`: `x`
*/
const sparsePlus = jit$1((x) => {
	return where(x.lessEqual(-1), 0, where(x.less(1), square(x.add(1)).mul(.25), x));
});
/**
* @function
* Sparse sigmoid activation function.
*
* - When `x <= -1`: `0`
* - When `-1 < x < 1`: `(x + 1) / 2`
* - When `x >= 1`: `1`
*/
const sparseSigmoid = jit$1((x) => {
	try {
		var _usingCtx4 = (0, import_usingCtx$1.default)();
		const shifted = _usingCtx4.u(x.add(1));
		const scaled = _usingCtx4.u(shifted.mul(.5));
		return clip(scaled, 0, 1);
	} catch (_) {
		_usingCtx4.e = _;
	} finally {
		_usingCtx4.d();
	}
});
/**
* Soft-sign activation function, computed element-wise:
* `softsign(x) = x / (|x| + 1)`.
*/
function softSign(x) {
	try {
		var _usingCtx5 = (0, import_usingCtx$1.default)();
		x = fudgeArray(x);
		const absX = _usingCtx5.u(absolute(x));
		const denom = _usingCtx5.u(absX.add(1));
		return x.div(denom);
	} catch (_) {
		_usingCtx5.e = _;
	} finally {
		_usingCtx5.d();
	}
}
/**
* @function
* Sigmoid-weighted Linear Unit (SiLU) activation function, also known as
* Swish, computed element-wise:
* `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`.
*
* `swish()` and `silu()` are both aliases for the same function.
*
* Reference: https://en.wikipedia.org/wiki/Swish_function
*/
const silu = jit$1(function silu$1(x) {
	return x.mul(sigmoid(x));
});
/**
* Log-sigmoid activation function, computed element-wise:
* `log_sigmoid(x) = log(sigmoid(x)) = -log(1 + exp(-x))`.
*/
function logSigmoid(x) {
	try {
		var _usingCtx6 = (0, import_usingCtx$1.default)();
		const neg$1 = _usingCtx6.u(negative(x));
		const sp = _usingCtx6.u(softplus(neg$1));
		return negative(sp);
	} catch (_) {
		_usingCtx6.e = _;
	} finally {
		_usingCtx6.d();
	}
}
/**
* @function
* Identity activation function. Returns the argument unmodified.
*/
const identity = fudgeArray;
/** Leaky rectified linear (ReLU) activation function */
function leakyRelu(x, negativeSlope = .01) {
	try {
		var _usingCtx7 = (0, import_usingCtx$1.default)();
		x = fudgeArray(x);
		const cond = _usingCtx7.u(less(x, 0));
		const scaled = _usingCtx7.u(x.mul(negativeSlope));
		return where(cond, scaled, x);
	} catch (_) {
		_usingCtx7.e = _;
	} finally {
		_usingCtx7.d();
	}
}
/** Hard sigmoid activation function: `relu6(x+3)/6`. */
function hardSigmoid(x) {
	try {
		var _usingCtx8 = (0, import_usingCtx$1.default)();
		const sum$1 = _usingCtx8.u(add(x, 3));
		const r = _usingCtx8.u(relu6(sum$1));
		return r.mul(1 / 6);
	} catch (_) {
		_usingCtx8.e = _;
	} finally {
		_usingCtx8.d();
	}
}
/** Hard SiLU (swish) activation function: `x * hardSigmoid(x)`. */
function hardSilu(x) {
	try {
		var _usingCtx9 = (0, import_usingCtx$1.default)();
		x = fudgeArray(x);
		const hs = _usingCtx9.u(hardSigmoid(x));
		return x.mul(hs);
	} catch (_) {
		_usingCtx9.e = _;
	} finally {
		_usingCtx9.d();
	}
}
/** Hard tanh activation function: `clip(x, -1, 1)`. */
function hardTanh(x) {
	return clip(x, -1, 1);
}
/**
* Exponential linear unit activation function.
*
* Computes the element-wise function:
* `elu(x) = x > 0 ? x : alpha * (exp(x) - 1)`
*/
function elu(x, alpha = 1) {
	try {
		var _usingCtx10 = (0, import_usingCtx$1.default)();
		x = fudgeArray(x);
		const cond = _usingCtx10.u(less(x, 0));
		const e$1 = _usingCtx10.u(exp(x));
		const em1 = _usingCtx10.u(e$1.sub(1));
		const scaled = _usingCtx10.u(em1.mul(alpha));
		return where(cond, scaled, x);
	} catch (_) {
		_usingCtx10.e = _;
	} finally {
		_usingCtx10.d();
	}
}
/**
* Continuously-differentiable exponential linear unit activation function.
*
* Computes the element-wise function:
* `celu(x) = x > 0 ? x : alpha * (exp(x/alpha) - 1)`
*/
function celu(x, alpha = 1) {
	try {
		var _usingCtx11 = (0, import_usingCtx$1.default)();
		x = fudgeArray(x);
		const cond = _usingCtx11.u(less(x, 0));
		const ratio = _usingCtx11.u(x.div(alpha));
		const e$1 = _usingCtx11.u(exp(ratio));
		const em1 = _usingCtx11.u(e$1.sub(1));
		const scaled = _usingCtx11.u(em1.mul(alpha));
		return where(cond, scaled, x);
	} catch (_) {
		_usingCtx11.e = _;
	} finally {
		_usingCtx11.d();
	}
}
/**
* @function
* Scaled exponential linear unit activation.
*
* Computes the element-wise function:
* `selu(x) = lambda * (x > 0 ? x : alpha * (exp(x) - 1))`
*
* Where `alpha = 1.6732632423543772` and `lambda = 1.0507009873554805`.
*/
const selu = jit$1(function selu$1(x) {
	const alpha = 1.6732632423543772;
	const lambda = 1.0507009873554805;
	return where(x.less(0), expm1(x).mul(alpha), x).mul(lambda);
});
/**
* @function
* Gaussion error linear unit (GELU) activation function.
*
* This is computed element-wise. There are two variants depending on whether
* `approximate` is set (default true):
*
* - Approximate: `gelu(x) ~= x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
* - Exact: `gelu(x) = x * 0.5 * erfc(-x / sqrt(2))`
*
* Reference: https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.gelu_approx.html
*/
const gelu = jit$1(function gelu$1(x, opts) {
	if (opts?.approximate ?? true) try {
		var _usingCtx12 = (0, import_usingCtx$1.default)();
		const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
		const x2 = _usingCtx12.u(x.mul(x));
		const quadTerm = _usingCtx12.u(x2.mul(.044715));
		const poly = _usingCtx12.u(quadTerm.add(1));
		const polyScaled = _usingCtx12.u(x.mul(poly));
		const tanhArg = _usingCtx12.u(polyScaled.mul(SQRT_2_OVER_PI));
		const tanhVal = _usingCtx12.u(tanh(tanhArg));
		const gate = _usingCtx12.u(tanhVal.add(1));
		const halfX = _usingCtx12.u(x.mul(.5));
		return halfX.mul(gate);
	} catch (_) {
		_usingCtx12.e = _;
	} finally {
		_usingCtx12.d();
	}
	else try {
		var _usingCtx13 = (0, import_usingCtx$1.default)();
		const scaled = _usingCtx13.u(x.mul(Math.SQRT1_2));
		const negScaled = _usingCtx13.u(negative(scaled));
		const erfComp = _usingCtx13.u(erfc$1(negScaled));
		const halfX = _usingCtx13.u(x.mul(.5));
		return halfX.mul(erfComp);
	} catch (_) {
		_usingCtx13.e = _;
	} finally {
		_usingCtx13.d();
	}
}, { staticArgnums: [1] });
/**
* Gated linear unit (GLU) activation function.
*
* Splits the `axis` dimension of the input into two halves, a and b, then
* computes `a * sigmoid(b)`.
*/
function glu(x, axis = -1) {
	try {
		var _usingCtx14 = (0, import_usingCtx$1.default)();
		x = fudgeArray(x);
		axis = require_backend.checkAxis(axis, x.ndim);
		const size$1 = x.shape[axis];
		if (size$1 % 2 !== 0) throw new Error(`glu: axis ${axis} of shape (${x.shape}) does not have even length`);
		const slice = x.shape.map((a$1) => [0, a$1]);
		const a = _usingCtx14.u(shrink(x, slice.toSpliced(axis, 1, [0, size$1 / 2])));
		const b = _usingCtx14.u(shrink(x, slice.toSpliced(axis, 1, [size$1 / 2, size$1])));
		const sig = _usingCtx14.u(sigmoid(b));
		return a.mul(sig);
	} catch (_) {
		_usingCtx14.e = _;
	} finally {
		_usingCtx14.d();
	}
}
/**
* Squareplus activation function.
*
* Computes the element-wise function:
* `squareplus(x) = 0.5 * (x + sqrt(x^2 + b))`
*/
function squareplus(x, b = 4) {
	try {
		var _usingCtx15 = (0, import_usingCtx$1.default)();
		x = fudgeArray(x);
		const sq = _usingCtx15.u(square(x));
		const sumSq = _usingCtx15.u(sq.add(b));
		const sr = _usingCtx15.u(sqrt(sumSq));
		const sum$1 = _usingCtx15.u(x.add(sr));
		return sum$1.mul(.5);
	} catch (_) {
		_usingCtx15.e = _;
	} finally {
		_usingCtx15.d();
	}
}
/**
* Mish activation function.
*
* Computes the element-wise function:
* `mish(x) = x * tanh(softplus(x))`
*/
function mish(x) {
	try {
		var _usingCtx16 = (0, import_usingCtx$1.default)();
		x = fudgeArray(x);
		const sp = _usingCtx16.u(softplus(x));
		const t = _usingCtx16.u(tanh(sp));
		return x.mul(t);
	} catch (_) {
		_usingCtx16.e = _;
	} finally {
		_usingCtx16.d();
	}
}
/**
* Softmax function. Computes the function which rescales elements to the range
* [0, 1] such that the elements along `axis` sum to 1.
*
* If `axis` is not specified, it defaults to the last axis.
*
* Reference: https://en.wikipedia.org/wiki/Softmax_function
*/
function softmax(x, axis = -1) {
	try {
		var _usingCtx17 = (0, import_usingCtx$1.default)();
		x = fudgeArray(x);
		axis = require_backend.normalizeAxis(axis, x.ndim);
		if (axis.length === 0) return onesLike(x);
		const xMax = _usingCtx17.u(max(x, axis, { keepdims: true }));
		const sg = stopGradient(xMax);
		const shifted = _usingCtx17.u(x.sub(sg));
		const unnormalized = _usingCtx17.u(exp(shifted));
		const denom = _usingCtx17.u(unnormalized.sum(axis, { keepdims: true }));
		return unnormalized.div(denom);
	} catch (_) {
		_usingCtx17.e = _;
	} finally {
		_usingCtx17.d();
	}
}
/**
* Log-Softmax function.
*
* Computes the logarithm of the `softmax` function, which rescales elements to
* the range [-infinity, 0).
*
* If `axis` is not specified, it defaults to the last axis.
*/
function logSoftmax(x, axis = -1) {
	try {
		var _usingCtx18 = (0, import_usingCtx$1.default)();
		x = fudgeArray(x);
		axis = require_backend.normalizeAxis(axis, x.ndim);
		if (axis.length === 0) return zerosLike(x);
		const xMax = _usingCtx18.u(max(x, axis, { keepdims: true }));
		const sg = stopGradient(xMax);
		const shifted = _usingCtx18.u(x.sub(sg));
		const expShifted = _usingCtx18.u(exp(shifted));
		const sumExp = _usingCtx18.u(expShifted.sum(axis, { keepdims: true }));
		const shiftedLogsumexp = _usingCtx18.u(log(sumExp));
		return shifted.sub(shiftedLogsumexp);
	} catch (_) {
		_usingCtx18.e = _;
	} finally {
		_usingCtx18.d();
	}
}
/**
* Log-sum-exp reduction. Also a multivariate version of `softplus`.
*
* If no axis is specified, the reduction is performed over all elements. This
* convention differs from `jax.nn.logSoftmax()`.
*
* Reference: https://en.wikipedia.org/wiki/LogSumExp
*/
function logsumexp(x, axis = null, opts) {
	try {
		var _usingCtx19 = (0, import_usingCtx$1.default)();
		x = fudgeArray(x);
		axis = require_backend.normalizeAxis(axis, x.ndim);
		if (axis.length === 0) return x;
		const rawMax = _usingCtx19.u(max(x, axis, { keepdims: true }));
		const xMax = stopGradient(rawMax);
		const shifted = _usingCtx19.u(x.sub(xMax));
		const expShifted = _usingCtx19.u(exp(shifted));
		const sumExp = _usingCtx19.u(expShifted.sum(axis, { keepdims: true }));
		const logSum = _usingCtx19.u(log(sumExp));
		if (opts?.keepdims) return xMax.add(logSum);
		const resultToSqueeze = _usingCtx19.u(xMax.add(logSum));
		return squeeze(resultToSqueeze, axis);
	} catch (_) {
		_usingCtx19.e = _;
	} finally {
		_usingCtx19.d();
	}
}
/** Log-mean-exp reduction, like `jax.nn.logsumexp()` but subtracts `log(n)`. */
function logmeanexp(x, axis = null, opts) {
	try {
		var _usingCtx20 = (0, import_usingCtx$1.default)();
		x = fudgeArray(x);
		axis = require_backend.normalizeAxis(axis, x.ndim);
		if (axis.length === 0) return x;
		const n = axis.reduce((acc, a) => acc * x.shape[a], 1);
		const lse = _usingCtx20.u(logsumexp(x, axis, opts));
		return lse.sub(Math.log(n));
	} catch (_) {
		_usingCtx20.e = _;
	} finally {
		_usingCtx20.d();
	}
}
/**
* Standardizes input to zero mean and unit variance.
*
* By default, this is computed over the last axis. You can pass in a different
* axis, or `null` to standardize over all elements.
*
* Epsilon is added to denominator, it defaults to `1e-5` for stability.
*/
function standardize(x, axis = -1, opts = {}) {
	x = fudgeArray(x);
	axis = require_backend.normalizeAxis(axis, x.ndim);
	if (axis.length === 0) return x;
	const d = [];
	try {
		const mu = opts.mean !== void 0 ? fudgeArray(opts.mean) : (() => {
			const m = x.mean(axis, { keepdims: true });
			d.push(m);
			return m;
		})();
		let sigma2;
		if (opts.variance !== void 0) sigma2 = fudgeArray(opts.variance);
		else {
			const sq = square(x);
			d.push(sq);
			const sqMean = sq.mean(axis, { keepdims: true });
			d.push(sqMean);
			const muSq = square(mu);
			d.push(muSq);
			sigma2 = sqMean.sub(muSq);
			d.push(sigma2);
		}
		const centered = x.sub(mu);
		d.push(centered);
		const denom = sigma2.add(opts.epsilon ?? 1e-5);
		d.push(denom);
		const sqrtDenom = sqrt(denom);
		d.push(sqrtDenom);
		return centered.div(sqrtDenom);
	} finally {
		for (const v of d) v[Symbol.dispose]();
	}
}
/**
* One-hot encodes the given indices.
*
* Each index in the integer input `x` is encoded as a vector of zeros of length
* `numClasses`, with a 1 at the index position specified by its value.
*
* ```js
* import { nn, numpy as np } from '@jax-js/jax';
*
* nn.oneHot(np.array([1, 1, 2], { dtype: np.int32 }), 3);
* // Output:
* // [[0, 1, 0],
* //  [0, 1, 0],
* //  [0, 0, 1]]
* ```
*/
function oneHot(x, numClasses) {
	if (require_backend.isFloatDtype(x.dtype)) throw new TypeError(`oneHot expects integers, got ${x.dtype}`);
	return eye(numClasses, void 0, { device: x.device }).slice(x);
}
/**
* Scaled dot product attention (SDPA).
*
* Computes `softmax((Q @ K^T) / sqrt(d) + bias) @ V`, where `Q` is the query,
* `K` is the key, `V` is the value, and `d` is the dimensionality of each key
* and query vector.
*
* Multi-query attention is applied when input `key` and `value` tensors have
* fewer heads than `query`.
*
* We use the following uppercase letters to denote array shapes:
* - `B` = batch size
* - `S` = length of key/value sequences (source)
* - `L` = length of query sequences
* - `N` = number of attention heads
* - `H` = dimensionality of each attention head
* - `K` = number of key/value heads (for grouped-query attention)
*
* The batch size `B` may be omitted, which is equivalent to `B = 1`. In this
* case it must be omitted from all inputs.
*
* @param query - Query array; shape `[B, L, N, H]`
* @param key - Key array; shape `[B, S, K, H]`
* @param value - Value array; same shape as `key`
* @param opts.bias - Optional bias to add to the attention logits; shape
*   `[B, N, L, S]` or broadcastable to it.
* @param opts.mask - Optional mask to apply to the attention logits; should be
*   a boolean array broadcastable to `[B, N, L, S]`, where `true` indicates
*   the element should take part in attention.
* @param opts.scale - Scaling factor override, default is `1 / sqrt(H)`.
* @param opts.isCausal - If true, applies a casual mask.
* @param opts.querySeqLengths - Optional sequence lengths for the queries;
*   shape `(B,)`. Taken from the beginning of the tensor.
* @param opts.keyValueSeqLengths - Optional sequence lengths for the keys and
*   values; shape `(B,)`. Taken from the beginning of the tensor.
* @param opts.localWindowSize - If specified, applies a local attention window
*   of the given size. Can be a single number or a tuple `[left, right]`.
*
* @returns The result of the attention operation; shape is the same as query
*   `[B, L, N, H]`, or `[L, N, H]` if `B` is omitted.
*/
function dotProductAttention(query, key$1, value, opts = {}) {
	query = fudgeArray(query);
	key$1 = fudgeArray(key$1);
	value = fudgeArray(value);
	if (query.ndim !== 3 && query.ndim !== 4 || query.ndim !== key$1.ndim || query.ndim !== value.ndim) throw new Error(`dotProductAttention: expected all tensors to have rank 3 or 4, got Q=${query.aval}, K=${key$1.aval}, V=${value.aval}`);
	if (!require_backend.deepEqual(key$1.shape, value.shape)) throw new Error(`dotProductAttention: key and value shapes must match, got K=${key$1.shape}, V=${value.shape}`);
	const isRank3 = query.ndim === 3;
	const d = [];
	try {
		if (isRank3) {
			query = expandDims(query, 0);
			d.push(query);
			key$1 = expandDims(key$1, 0);
			d.push(key$1);
			value = expandDims(value, 0);
			d.push(value);
		}
		const [B, L, N, H] = query.shape;
		if (key$1.shape[0] !== B || key$1.shape[3] !== H) throw new Error(`dotProductAttention: query and key shapes mismatch, got Q=${query.aval}, K=${key$1.aval}`);
		const S = key$1.shape[1];
		const K = key$1.shape[2];
		if (N < K || N != K && N % K !== 0) throw new Error(`dotProductAttention: number of query heads N=${N} must be divisible by number of key/value heads K=${K} for GQA`);
		const G = N / K;
		key$1 = tile(key$1, [
			1,
			1,
			G,
			1
		]);
		d.push(key$1);
		value = tile(value, [
			1,
			1,
			G,
			1
		]);
		d.push(value);
		const scale = opts.scale ?? 1 / Math.sqrt(H);
		const rawScores = einsum("BLNH,BSNH->BNLS", query, key$1);
		d.push(rawScores);
		let scores = rawScores.mul(scale);
		if (opts.bias !== void 0) {
			d.push(scores);
			scores = scores.add(opts.bias);
		}
		if (opts.mask !== void 0) {
			d.push(scores);
			scores = where(opts.mask, scores, -Infinity);
		}
		if (opts.isCausal) {
			const causalMask = tri(L, S, 0, { dtype: require_backend.DType.Bool });
			d.push(causalMask);
			d.push(scores);
			scores = where(causalMask, scores, -Infinity);
		}
		if (opts.localWindowSize !== void 0) {
			const [before, after] = typeof opts.localWindowSize === "number" ? [opts.localWindowSize, opts.localWindowSize] : opts.localWindowSize;
			if (before < 0 || after < 0 || !Number.isInteger(before) || !Number.isInteger(after)) throw new Error(`dotProductAttention: localWindowSize values must be non-negative, got ${opts.localWindowSize}`);
			const triAfter = tri(L, S, after, { dtype: require_backend.DType.Bool });
			d.push(triAfter);
			const triBefore = tri(L, S, -before - 1, { dtype: require_backend.DType.Bool });
			d.push(triBefore);
			const triBeforeNot = triBefore.notEqual(true);
			d.push(triBeforeNot);
			const localMask = triAfter.mul(triBeforeNot);
			d.push(localMask);
			d.push(scores);
			scores = where(localMask, scores, -Infinity);
		}
		if (opts.querySeqLengths !== void 0) {
			const sl = expandDims(opts.querySeqLengths, [
				-1,
				-2,
				-3
			]);
			d.push(sl);
			const ar = arange(L);
			d.push(ar);
			const arR = ar.reshape([
				1,
				1,
				L,
				1
			]);
			d.push(arR);
			const cond = arR.less(sl);
			d.push(cond);
			d.push(scores);
			scores = where(cond, scores, -Infinity);
		}
		if (opts.keyValueSeqLengths !== void 0) {
			const sl = expandDims(opts.keyValueSeqLengths, [
				-1,
				-2,
				-3
			]);
			d.push(sl);
			const ar = arange(S);
			d.push(ar);
			const arR = ar.reshape([
				1,
				1,
				1,
				S
			]);
			d.push(arR);
			const cond = arR.less(sl);
			d.push(cond);
			d.push(scores);
			scores = where(cond, scores, -Infinity);
		}
		d.push(scores);
		const attn = softmax(scores, -1);
		d.push(attn);
		const out = einsum("BNLS,BSNH->BLNH", attn, value);
		if (isRank3) {
			d.push(out);
			return out.reshape([
				L,
				N,
				H
			]);
		}
		return out;
	} finally {
		for (const v of d) v[Symbol.dispose]();
	}
}

//#endregion
//#region src/library/random.ts
var random_exports = {};
__export(random_exports, {
	bernoulli: () => bernoulli,
	bits: () => bits,
	categorical: () => categorical,
	cauchy: () => cauchy,
	exponential: () => exponential,
	gumbel: () => gumbel,
	key: () => key,
	laplace: () => laplace,
	multivariateNormal: () => multivariateNormal,
	normal: () => normal,
	split: () => split,
	uniform: () => uniform
});
var import_usingCtx = /* @__PURE__ */ __toESM(require_usingCtx(), 1);
function validateKeyShape(key$1, scalar = false) {
	if (key$1.ndim === 0) throw new Error("Key must have at least one dimension.");
	if (key$1.shape[key$1.shape.length - 1] !== 2) throw new Error(`Invalid key shape: ${key$1.shape}. Expected last dimension to be 2.`);
	if (scalar && key$1.shape.length > 1) throw new Error(`Expected a single PRNG key, but got a batch of keys with shape ${JSON.stringify(key$1.shape)} - use jax.vmap for batching.`);
	return key$1.shape.slice(0, -1);
}
function getK01(key$1) {
	const keyShape = validateKeyShape(key$1, true);
	const splits = split$2(key$1, -1, [1, 1]);
	const k0 = splits[0].reshape(keyShape);
	const k1 = splits[1].reshape(keyShape);
	splits[0].dispose();
	splits[1].dispose();
	return [k0, k1];
}
/** Create a pseudo-random number generator (PRNG) key from 32-bit integer seed. */
function key(seed) {
	try {
		var _usingCtx$1 = (0, import_usingCtx.default)();
		const seedArr = _usingCtx$1.u(array(seed, { dtype: require_backend.DType.Uint32 }));
		if (seedArr.ndim !== 0) throw new Error(`key: seed must be a scalar integer, but got shape ${seedArr.shape} - use jax.vmap for batching.`);
		const key$1 = stack([0, seedArr]);
		if (key$1 instanceof Array$1) key$1._realizeSource();
		return key$1;
	} catch (_) {
		_usingCtx$1.e = _;
	} finally {
		_usingCtx$1.d();
	}
}
/** Splits a PRNG key into `num` new keys by adding a leading axis. */
function split(key$1, num = 2) {
	const shape$1 = typeof num === "number" ? [num] : num;
	for (const len of shape$1) if (len <= 0 || !Number.isInteger(len)) throw new Error(`Invalid split length: ${len}. Must be a positive integer.`);
	const [k0, k1] = getK01(key$1);
	try {
		try {
			var _usingCtx3 = (0, import_usingCtx.default)();
			const r0 = _usingCtx3.u(randomBits(k0, k1, shape$1, 0));
			const r1 = _usingCtx3.u(randomBits(k0, k1, shape$1, 1));
			return stack([r0, r1], -1);
		} catch (_) {
			_usingCtx3.e = _;
		} finally {
			_usingCtx3.d();
		}
	} finally {
		k0.dispose();
		k1.dispose();
	}
}
/** Sample uniform bits in the form of unsigned integers. */
function bits(key$1, shape$1 = []) {
	const [k0, k1] = getK01(key$1);
	try {
		return randomBits(k0, k1, shape$1);
	} finally {
		k0.dispose();
		k1.dispose();
	}
}
/**
* @function
* Sample uniform random values in [minval, maxval) with given shape.
*/
const uniform = jit$1(function uniform$1(key$1, shape$1 = [], { minval = 0, maxval = 1 } = {}) {
	try {
		var _usingCtx4 = (0, import_usingCtx.default)();
		if (minval >= maxval) throw new Error(`Invalid range: [${minval}, ${maxval}).`);
		const divisor = _usingCtx4.u(array(512, {
			dtype: require_backend.DType.Uint32,
			device: key$1.device
		}));
		const bias = _usingCtx4.u(array(1065353216, {
			dtype: require_backend.DType.Uint32,
			device: key$1.device
		}));
		const mantissa = _usingCtx4.u(bits(key$1, shape$1).div(divisor));
		const float12 = _usingCtx4.u(mantissa.add(bias));
		const rand = bitcast(float12, require_backend.DType.Float32).sub(1);
		if (minval === 0 && maxval === 1) return rand;
		else try {
			var _usingCtx5 = (0, import_usingCtx.default)();
			const scaled = _usingCtx5.u(rand.mul(maxval - minval));
			return scaled.add(minval);
		} catch (_) {
			_usingCtx5.e = _;
		} finally {
			_usingCtx5.d();
		}
	} catch (_) {
		_usingCtx4.e = _;
	} finally {
		_usingCtx4.d();
	}
}, { staticArgnums: [1, 2] });
/**
* Sample Bernoulli random variables with given mean (0,1 categorical).
*
* Returns a random Boolean array with the specified shape. `p` can be an array
* and must be broadcastable to `shape`.
*/
function bernoulli(key$1, p = .5, shape$1 = []) {
	try {
		var _usingCtx6 = (0, import_usingCtx.default)();
		p = fudgeArray(p);
		const u = _usingCtx6.u(uniform(key$1, shape$1));
		return u.less(p);
	} catch (_) {
		_usingCtx6.e = _;
	} finally {
		_usingCtx6.d();
	}
}
/**
* @function
* Sample random values from categorical distributions.
*
* Uses the Gumbel max trick for sampling with replacement, or the Gumbel top-k
* trick for sampling without replacement.
*
* Note: Sampling without replacement currently uses argsort and slices the last
* k elements. This should be replaced with a more efficient topK implementation.
*
* - `key` - PRNG key
* - `logits` - Unnormalized log probabilities of the categorical distribution(s).
*   `softmax(logits, axis)` gives the corresponding probabilities.
* - `axis` - Axis along which logits belong to the same categorical distribution.
* - `shape` - Result batch shape. Must be broadcast-compatible with
*   `logits.shape` with `axis` removed. Default is `logits.shape` with `axis` removed.
* - `replace` - If true (default), sample with replacement. If false, sample
*   without replacement (each category can only be selected once per batch).
* @returns A random array with int dtype and shape given by `shape` if provided,
*   otherwise `logits.shape` with `axis` removed.
*/
const categorical = jit$1(function categorical$1(key$1, logits, { axis = -1, shape: shape$1, replace = true } = {}) {
	logits = fudgeArray(logits);
	axis = require_backend.checkAxis(axis, logits.ndim);
	const numCategories = logits.shape[axis];
	const batchShape = logits.shape.toSpliced(axis, 1);
	if (shape$1 === void 0) shape$1 = batchShape;
	else if (!require_backend.deepEqual(require_backend.generalBroadcast(shape$1, batchShape), shape$1)) throw new Error(`Shape ${shape$1} is not broadcast-compatible with batch shape ${batchShape}.`);
	const shapePrefix = shape$1.slice(0, shape$1.length - batchShape.length);
	if (replace) {
		const noise = gumbel(key$1, [...shapePrefix, ...logits.shape]);
		return argmax(noise.add(logits), axis + shapePrefix.length);
	} else {
		const k = shapePrefix.reduce((a, b) => a * b, 1);
		if (k > numCategories) throw new Error(`Number of samples without replacement (${k}) cannot exceed number of categories (${numCategories}).`);
		const noise = gumbel(key$1, logits.shape);
		const [values, indices] = topK(noise.add(logits), k, axis);
		values.dispose();
		return indices.reshape(shape$1);
	}
}, { staticArgnums: [2] });
/**
* @function
* Sample from a Cauchy distribution with location 0 and scale 1.
*
* Uses inverse transform sampling: `x = tan(π * (u - 0.5))` where u ~ Uniform(0, 1).
*/
const cauchy = jit$1(function cauchy$1(key$1, shape$1 = []) {
	try {
		var _usingCtx7 = (0, import_usingCtx.default)();
		const u = uniform(key$1, shape$1);
		const centered = _usingCtx7.u(u.sub(.5));
		const scaled = _usingCtx7.u(centered.mul(Math.PI));
		return tan(scaled);
	} catch (_) {
		_usingCtx7.e = _;
	} finally {
		_usingCtx7.d();
	}
}, { staticArgnums: [1] });
/**
* @function
* Sample exponential random values according to `p(x) = exp(-x)`.
*/
const exponential = jit$1(function exponential$1(key$1, shape$1 = []) {
	const u = uniform(key$1, shape$1);
	return negative(log1p(negative(u)));
}, { staticArgnums: [1] });
/**
* @function
* Sample from a Gumbel distribution with location 0 and scale 1.
*
* Uses inverse transform sampling: `x = -log(-log(u))` where u ~ Uniform(0, 1).
*/
const gumbel = jit$1(function gumbel$1(key$1, shape$1 = []) {
	const u = uniform(key$1, shape$1);
	return negative(log(negative(log1p(negative(u)))));
}, { staticArgnums: [1] });
/**
* @function
* Sample from a Laplace distribution with location 0 and scale 1.
*
* Uses inverse transform sampling: the CDF is `F(x) = 0.5 + 0.5 * sign(x) * (1 - exp(-|x|))`.
* Inverting: `x = -sign(u - 0.5) * log(1 - 2 * |u - 0.5|)`.
*/
const laplace = jit$1(function laplace$1(key$1, shape$1 = []) {
	try {
		var _usingCtx8 = (0, import_usingCtx.default)();
		const u = uniform(key$1, shape$1);
		const centered = _usingCtx8.u(u.sub(.5));
		const s = sign(centered);
		const absVal = absolute(centered);
		return s.mul(log1p(absVal.mul(-2)).mul(-1));
	} catch (_) {
		_usingCtx8.e = _;
	} finally {
		_usingCtx8.d();
	}
}, { staticArgnums: [1] });
/**
* @function
* Sample multivariate normal random values with given mean and covariance.
*
* The values are returned with the given shape, along with the final dimension
* used to represent the n-dimensional multivariate normal factors.
*
* This uses Cholesky decomposition on the covariance matrix.
*
* - `key` - PRNG key
* - `mean` - Mean vector of shape `[..., n]`
* - `cov` - Covariance of shape `[..., n, n]`, must be positive-definite
* - `shape` - Result batch shape, must be broadcastable with
*            `mean.shape[:-1]` and `cov.shape[:-2]`
* @returns Random samples of shape `[...shape, n]`
*/
const multivariateNormal = jit$1(function multivariateNormal$1(key$1, mean$1, cov$1, shape$1 = []) {
	mean$1 = fudgeArray(mean$1);
	cov$1 = fudgeArray(cov$1);
	const n = mean$1.shape[mean$1.ndim - 1];
	if (cov$1.shape[cov$1.ndim - 1] !== n || cov$1.shape[cov$1.ndim - 2] !== n) throw new Error(`Invalid covariance shape: ${cov$1.shape}. Expected last two dimensions to be [${n}, ${n}].`);
	const outputShape = broadcastShapes(shape$1, mean$1.shape.slice(0, -1), cov$1.shape.slice(0, -2)).concat(n);
	const L = cholesky(cov$1);
	const z = normal(key$1, outputShape);
	return einsum("...ij,...j->...i", L, z).add(mean$1);
}, { staticArgnums: [3] });
/**
* @function
* Sample random values according to `p(x) = 1/sqrt(2pi) * exp(-x^2/2)`.
*
* Unlike JAX, this uses the Box-Muller transform. JAX uses the erf_inv primitive instead and
* directly inverts the CDF, but we don't have support for that yet. Outputs will not be
* bitwise identical to JAX.
*/
const normal = jit$1(function normal$1(key$1, shape$1 = []) {
	const [k1, k2] = split(key$1, 2);
	const u1 = uniform(k1, shape$1);
	const u2 = uniform(k2, shape$1);
	const radius = sqrt(log1p(negative(u1)).mul(-2));
	const theta = u2.mul(2 * Math.PI);
	return radius.mul(cos(theta));
}, { staticArgnums: [1] });

//#endregion
//#region src/library/scipy-special.ts
var scipy_special_exports = {};
__export(scipy_special_exports, {
	erf: () => erf,
	erfc: () => erfc,
	logSoftmax: () => logSoftmax,
	logit: () => logit,
	logsumexp: () => logsumexp,
	softmax: () => softmax
});
/**
* @function
* The logit function, `logit(p) = log(p / (1-p))`.
*/
const logit = jit$1(function logit$1(x) {
	return log(x.div(subtract(1, x)));
});

//#endregion
//#region src/polyfills.ts
/** @file Polyfills for using this library. */
Symbol.dispose ??= Symbol.for("Symbol.dispose");
Symbol.asyncDispose ??= Symbol.for("Symbol.asyncDispose");

//#endregion
//#region src/index.ts
/**
* @function
* Compute the forward-mode Jacobian-vector product for a function.
*/
const jvp = jvp$1;
/**
* @function
* Vectorize an operation on a batched axis for one or more inputs.
*/
const vmap = vmap$1;
/**
* @function
* Compute the Jacobian evaluated column-by-column by forward-mode AD.
*/
const jacfwd = jacfwd$1;
/**
* @function
* Construct a Jaxpr by dynamically tracing a function with example inputs.
*/
const makeJaxpr = makeJaxpr$1;
/**
* @function
* Mark a function for automatic JIT compilation, with operator fusion.
*
* The function will be compiled the first time it is called with a set of
* argument shapes.
*
* You can call `.dispose()` on the returned, JIT-compiled function after all
* calls to free memory associated with array constants.
*
* **Options:**
* - `staticArgnums`: An array of argument indices to treat as static
*   (compile-time constant). These arguments must be hashable, won't be traced,
*   and different values will trigger recompilation.
* - `device`: The device to place the computation on. If not specified, the
*   computation will be placed on the default device.
*/
const jit = jit$1;
/**
* @function
* Produce a local linear approximation to a function at a point using jvp() and
* partial evaluation.
*/
const linearize = linearize$1;
/**
* @function
* Calculate the reverse-mode vector-Jacobian product for a function.
*
* The return value is a tuple of `[out, vjpFn]`, where `out` is the output of
* `f(primals)`, and `vjpFn` is a function that takes in cotangents for each
* output and returns the cotangents for each input.
*
* When `{ hasAux: true }` is passed, the function `f` is expected to return an
* `[out, aux]` tuple, and `vjp` returns `[out, vjpFn, aux]`.
*
* @example
* ```ts
* const [y, vjpFn] = vjp(f, [x]);
*
* // With hasAux
* const [y, vjpFn, aux] = vjp(f, [x], { hasAux: true });
* ```
*/
const vjp = vjp$1;
/**
* @function
* Compute the gradient of a scalar-valued function `f` with respect to its
* first argument.
*
* Pass in different `argnums` to differentiate with respect to other
* arguments. If a tuple is provided, the return value will be a tuple of
* gradients corresponding to each argument index.
*
* When `{ hasAux: true }` is passed, the function `f` is expected to return a
* `[out, aux]` tuple, and the return value will be `[gradient, aux]`.
*
* @example
* ```ts
* const gradient = grad(f)(x);
*
* // With `argnums`
* const [gradientX, gradientZ] = grad(f, { argnums: [0, 2] })(x, y, z);
*
* // With `hasAux`
* const [gradient, aux] = grad(f, { hasAux: true })(x);
* ```
*/
const grad = grad$1;
/**
* @function
* Create a function that evaluates both `f` and the gradient of `f`.
*
* When `{ hasAux: true }` is passed, the function `f` is expected to return an
* `[out, aux]` tuple, and the return value will be `[[out, aux], gradient]`.
*
* @example
* ```ts
* // Without hasAux
* const [value, gradient] = valueAndGrad(f)(x);
*
* // With hasAux
* const [[value, aux], gradient] = valueAndGrad(f, { hasAux: true })(x);
* ```
*/
const valueAndGrad = valueAndGrad$1;
/**
* @function
* Compute the Jacobian evaluated row-by-row by reverse-mode AD.
*/
const jacrev = jacrev$1;
/**
* @function
* Compute the Hessian matrix of a scalar-valued function.
*
* The Hessian is the matrix of second-order partial derivatives of a function.
* This is implemented as `jacfwd(grad(f))`.
*
* @example
* ```ts
* const f = (x: np.Array) => np.sum(x.mul(x).mul(x)); // x^3
* const H = hessian(f)(np.array([1, 2, 3]));
* // H[i,j] = d^2f / dx_i dx_j
* ```
*/
const hessian = hessian$1;
/**
* Wait until all `Array` leaves are ready by calling `Array.blockUntilReady()`.
*
* This can be used to wait for the results of an intermediate computation to
* finish. It's recommended to call this regularly in an iterative computation
* to avoid queueing up too many pending operations.
*
* Does not consume reference to the arrays.
*/
async function blockUntilReady(x) {
	const promises = [];
	for (const leaf of leaves(x)) if (leaf instanceof Array$1) promises.push(leaf.blockUntilReady());
	await Promise.all(promises);
	return x;
}
/**
* Transfer `x` to `device`.
*
* `x` may be a nested container of arrays or scalars. The resulting structure
* is committed to the device.
*
* If `device` is not specified, this function behaves as identity if the input
* is already an `Array`, otherwise it places the scalar uncommitted on the
* default device.
*/
async function devicePut(x, device) {
	const [xflat, structure$1] = flatten(x);
	const yflat = await Promise.all(xflat.map((leaf) => {
		if (leaf instanceof Array$1) return device ? leaf._put(require_backend.getBackend(device)) : Promise.resolve(leaf);
		else return Promise.resolve(array(leaf, { device }));
	}));
	return unflatten(structure$1, yflat);
}

//#endregion
exports.Array = Array$1;
exports.ClosedJaxpr = ClosedJaxpr;
exports.DType = require_backend.DType;
exports.Jaxpr = Jaxpr;
exports.blockUntilReady = blockUntilReady;
exports.checkLeaks = checkLeaks;
exports.defaultDevice = require_backend.defaultDevice;
exports.devicePut = devicePut;
exports.devices = require_backend.devices;
exports.getBackend = require_backend.getBackend;
exports.grad = grad;
exports.hessian = hessian;
exports.init = require_backend.init;
exports.jacfwd = jacfwd;
exports.jacobian = jacrev;
exports.jacrev = jacrev;
exports.jit = jit;
exports.jvp = jvp;
Object.defineProperty(exports, 'lax', {
  enumerable: true,
  get: function () {
    return lax_exports;
  }
});
exports.linearize = linearize;
exports.makeJaxpr = makeJaxpr;
Object.defineProperty(exports, 'nn', {
  enumerable: true,
  get: function () {
    return nn_exports;
  }
});
Object.defineProperty(exports, 'numpy', {
  enumerable: true,
  get: function () {
    return numpy_exports;
  }
});
Object.defineProperty(exports, 'random', {
  enumerable: true,
  get: function () {
    return random_exports;
  }
});
Object.defineProperty(exports, 'scipySpecial', {
  enumerable: true,
  get: function () {
    return scipy_special_exports;
  }
});
exports.setDebug = require_backend.setDebug;
Object.defineProperty(exports, 'tree', {
  enumerable: true,
  get: function () {
    return tree_exports;
  }
});
exports.valueAndGrad = valueAndGrad;
exports.vjp = vjp;
exports.vmap = vmap;