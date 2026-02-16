<script lang="ts">
  import mergeDescriptors from "merge-descriptors";
  import type * as Monaco from "monaco-editor/esm/vs/editor/editor.api.js";
  import { onDestroy, onMount } from "svelte";

  let {
    initialText,
    editorOptions = {},
    onchange,
    onformat,
    onrun,
  }: {
    initialText: string;
    editorOptions?: Monaco.editor.IStandaloneEditorConstructionOptions;
    onchange?: () => void;
    onformat?: () => void;
    onrun?: () => void;
  } = $props();

  let containerEl: HTMLDivElement;
  let editor: Monaco.editor.IStandaloneCodeEditor;
  let monaco: typeof Monaco;

  export function getText() {
    return editor?.getValue() ?? "";
  }

  export function setText(text: string) {
    editor?.setValue(text);
  }

  export function getCursorOffset() {
    const model = editor?.getModel();
    if (editor && model) {
      return model.getOffsetAt(
        editor.getPosition() ?? {
          lineNumber: 1,
          column: 1,
        },
      );
    } else {
      return 0;
    }
  }

  export function setCursorOffset(offset: number) {
    const model = editor?.getModel();
    if (editor && model) {
      const position = model.getPositionAt(offset);
      editor.setPosition(position);
    }
  }

  export function setLeakMarkers(markers: { line: number; message: string }[]) {
    if (!monaco || !editor) return;
    const model = editor.getModel();
    if (!model) return;
    monaco.editor.setModelMarkers(
      model,
      "leak-diagnostics",
      markers.map((m) => ({
        startLineNumber: m.line,
        startColumn: 1,
        endLineNumber: m.line,
        endColumn: model.getLineMaxColumn(m.line),
        message: m.message,
        severity: monaco.MarkerSeverity.Warning,
      })),
    );
  }

  onMount(async () => {
    monaco = (await import("$lib/monaco")).default;

    editor = monaco.editor.create(
      containerEl,
      mergeDescriptors(
        {
          fontSize: 13,
          automaticLayout: true,
          renderLineHighlight: "none",
          minimap: { enabled: false },
        },
        editorOptions,
      ),
    );
    if (onchange) {
      editor.onDidChangeModelContent(() => {
        onchange();
      });
    }
    editor.addAction({
      id: "format",
      label: "Format with Prettier",
      keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS],
      contextMenuGroupId: "navigation",
      contextMenuOrder: 1,
      run: () => {
        onformat?.();
      },
    });
    editor.addAction({
      id: "run",
      label: "Run Code",
      keybindings: [
        monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter,
        monaco.KeyMod.Shift | monaco.KeyCode.Enter,
      ],
      contextMenuGroupId: "navigation",
      contextMenuOrder: 0.9,
      run: () => {
        onrun?.();
      },
    });
    const model = monaco.editor.createModel(
      initialText,
      "typescript",
      monaco.Uri.parse("file:///main.ts"),
    );
    model.updateOptions({ tabSize: 2 });
    editor.setModel(model);
  });

  onDestroy(() => {
    monaco?.editor.getModels().forEach((model) => model.dispose());
    editor?.dispose();
  });
</script>

<div class="h-full" bind:this={containerEl}></div>

<style lang="postcss">
  @reference "$app.css";

  /* Cannot be disabled in config, see https://github.com/microsoft/vscode/issues/239240 */
  div :global(.iPadShowKeyboard) {
    display: none;
  }
</style>
