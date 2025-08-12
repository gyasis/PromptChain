import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Type helper for Svelte elements with ref
export type WithElementRef<T extends Record<string, any>> = T & {
  ref?: HTMLElement | null;
};

// Type helper to exclude children from component props
export type WithoutChildren<T> = Omit<T, "children">;