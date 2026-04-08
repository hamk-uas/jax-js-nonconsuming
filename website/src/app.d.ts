// See https://svelte.dev/docs/kit/types#app.d.ts
// for information about these interfaces
declare module "*?raw" {
  const source: string;
  export default source;
}

declare global {
  namespace App {
    // interface Error {}
    // interface Locals {}
    // interface PageData {}
    // interface PageState {}
    // interface Platform {}
  }
}

export {};
