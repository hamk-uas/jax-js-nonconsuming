import { resolve } from "$app/paths";
import { redirect } from "@sveltejs/kit";

import type { PageLoad } from "./$types";

export const load: PageLoad = () => {
  redirect(308, resolve(`tts`));
};
