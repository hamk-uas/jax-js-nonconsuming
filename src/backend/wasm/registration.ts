export interface AsyncModuleRegistrationTracker {
  ownerLabel: string;
  moduleIds: WeakMap<WebAssembly.Module, number>;
  registeringModules: Map<number, Promise<void>>;
  allocateModuleId: () => number;
  isRegistered: (id: number) => boolean;
  markRegistered: (id: number) => void;
  clearRegistered: (id: number) => void;
}

export interface AsyncModuleRegistrationResult {
  id: number;
  created: boolean;
}

export async function registerAsyncModule(
  tracker: AsyncModuleRegistrationTracker,
  module: WebAssembly.Module,
  startRegistration: (id: number) => Promise<void>,
  onRegistrationFailure: (id: number) => void | Promise<void>,
): Promise<AsyncModuleRegistrationResult> {
  const existing = tracker.moduleIds.get(module);
  if (existing !== undefined) {
    const pending = tracker.registeringModules.get(existing);
    if (pending) {
      await pending;
    } else if (!tracker.isRegistered(existing)) {
      throw new Error(
        `${tracker.ownerLabel}: module ${existing} is not registered and has no in-flight registration`,
      );
    }
    return { id: existing, created: false };
  }

  const id = tracker.allocateModuleId();
  tracker.moduleIds.set(module, id);

  const registration = startRegistration(id).then(() => {
    tracker.markRegistered(id);
  });
  tracker.registeringModules.set(id, registration);

  try {
    await registration;
    return { id, created: true };
  } catch (error) {
    tracker.moduleIds.delete(module);
    tracker.clearRegistered(id);
    await onRegistrationFailure(id);
    throw error;
  } finally {
    tracker.registeringModules.delete(id);
  }
}
