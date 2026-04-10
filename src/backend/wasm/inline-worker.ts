export function withInlineModuleWorkerUrl<T>(
  code: string,
  useUrl: (url: string) => T,
): T {
  const blob = new Blob([code], { type: "text/javascript" });
  const url = URL.createObjectURL(blob);
  try {
    return useUrl(url);
  } finally {
    // Defer revocation because module workers may not have loaded the blob URL
    // synchronously by the time revokeObjectURL is called.
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }
}
