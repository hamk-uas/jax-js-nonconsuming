const fs = require("fs");
let code = fs.readFileSync("src/backend/webgpu.ts", "utf8");

// The best way to defer EVERYTHING is simply to pass the bare WebGPUCommandTape
// and its arguments to PendingCommandTape, and let submit() run the entirety of
// executeCommandTape's inner logic, BUT we must return outputSlots synchronously!
