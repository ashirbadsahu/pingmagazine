import { defineCollection, z } from 'astro:content';

const magazineCollection = defineCollection({
    type: 'content',
    schema: ({ image }) => z.object({
        title: z.string(),
        publishDate: z.coerce.date().optional(),
        // Astro will optimize this image for fast loading
        cover: image(),
        // The path to your PDF in the /public/issues/ folder
        pdfUrl: z.string(),
    }),
});

export const collections = {
    'magazines': magazineCollection,
};