#include <locale>
#include "../../Application/TWAssert.h"
#include "ImageLoader.h"

optix::TextureSampler ImageLoader::LoadImageTexture(optix::Context & context, const std::string filename, const optix::float4 defaultColor, bool isOpacityMap)
{
	if (isOpacityMap)
		return LoadImageTextureAlpha(context, filename, defaultColor);

	// Get the filename extension                                                
	std::string::size_type extension_index = filename.find_last_of(".");
	std::string ext = extension_index != std::string::npos ?
		filename.substr(extension_index + 1) :
		std::string();
	std::locale loc;
	for (std::string::size_type i = 0; i < ext.length(); ++i)
		ext[i] = std::tolower(ext[i], loc);
	
	if (ext == "hdr" || ext == "exr")
		return LoadImageTextureLinear(context, filename, defaultColor);
	return LoadImageTextureGamma(context, filename, defaultColor);
}

void ImageLoader::SaveOutputBufferToImage(optix::Buffer buffer, const std::string filename)
{
	//////////////////////////////////////////////////////////////////////////
	//Load Image using FreeImage

	// If we're here we have a known image format, so load the image into a bitap
	RTsize bufferSizeW, bufferSizeH;
	buffer->getSize(bufferSizeW, bufferSizeH);
	FIBITMAP* bitmap = FreeImage_AllocateT(FIT_RGBAF, bufferSizeW, bufferSizeH);

	//--Debug info
	// How many bits-per-pixel is the source image?
	int bitsPerPixel = FreeImage_GetBPP(bitmap);

	int imageWidth = FreeImage_GetWidth(bitmap);
	int imageHeight = FreeImage_GetHeight(bitmap);

	FREE_IMAGE_TYPE imageType = FreeImage_GetImageType(bitmap);
	int bytespp = FreeImage_GetLine(bitmap) / imageWidth / sizeof(float);

	std::cout << "Image: " << filename << " is size: " << imageWidth << "x" << imageHeight << "with " << bitsPerPixel << "bits per pixel" << "." << std::endl;
	std::cout << "Image Type: " << imageType << std::endl;
	std::cout << "Image component(which is used to step to the next pixel):" << bytespp << std::endl;

	//--Debug info end

	//////////////////////////////////////////////////////////////////////////
	//Create Buffer
	float* buffer_data = static_cast<float*>(buffer->map());

	//todo:review BITMAP Scanline is upside down,see also FreeImage Pixel Access Function for further details.

	/* Note that we need to convert the data in sysOutputBuffer to linear,
	 * due to the "degamma" operation applied in Filtering phase to show
	 * in RenderView. 
	 * see also:https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_sRGB_decode.txt */
	auto convert_sRGBToLinear = [](const float &src)->float
	{
		if (src <= 0.f)
			return 0;
		if (src >= 1.f)
			return 1.f;
		if (src <= 0.04045f)
			return src / 12.92f;
		return pow((src + 0.055f) / 1.055f, 2.4f);
	};

	for (auto y = 0; y < imageHeight; ++y)
	{
		//Note the scanline fetched by FreeImage is upside down--the first scanline corresponds to the buttom of the image!
		FLOAT *bits = reinterpret_cast<FLOAT *>(FreeImage_GetScanLine(bitmap, /*imageHeight - y - 1*/y));

		for (auto x = 0; x < imageWidth; ++x)
		{
			unsigned int buf_index = (imageWidth * y + x) * 4;

			//32bit image texture is linear.
			//Todo:validate the input color value, in case of being nan/inf
			TW_ASSERT(!isinf(buffer_data[buf_index]) && !isnan(buffer_data[buf_index]));
			TW_ASSERT(!isinf(buffer_data[buf_index + 1]) && !isnan(buffer_data[buf_index + 1]));
			TW_ASSERT(!isinf(buffer_data[buf_index + 2]) && !isnan(buffer_data[buf_index + 2]));


			static_cast<float>(bits[0]) = convert_sRGBToLinear(buffer_data[buf_index]);
			static_cast<float>(bits[1]) = convert_sRGBToLinear(buffer_data[buf_index + 1]);
			static_cast<float>(bits[2]) = convert_sRGBToLinear(buffer_data[buf_index + 2]);
			static_cast<float>(bits[3]) = 1.f;
			// jump to next pixel
			bits += bytespp;
		}
	}

	buffer->unmap();
	FreeImage_Save(FIF_EXR, bitmap, filename.c_str());
	// Unload the 32-bit colour bitmap
	FreeImage_Unload(bitmap);
}

void ImageLoader::saveHDRBufferToImage(optix::Buffer buffer, const std::string & filename)
{
    //////////////////////////////////////////////////////////////////////////
    //Load Image using FreeImage

    // If we're here we have a known image format, so load the image into a bitap
    RTsize bufferSizeW, bufferSizeH;
    buffer->getSize(bufferSizeW, bufferSizeH);
    FIBITMAP* bitmap = FreeImage_AllocateT(FIT_RGBAF, bufferSizeW, bufferSizeH);

    //--Debug info
    // How many bits-per-pixel is the source image?
    int bitsPerPixel = FreeImage_GetBPP(bitmap);

    int imageWidth = FreeImage_GetWidth(bitmap);
    int imageHeight = FreeImage_GetHeight(bitmap);

    FREE_IMAGE_TYPE imageType = FreeImage_GetImageType(bitmap);
    int bytespp = FreeImage_GetLine(bitmap) / imageWidth / sizeof(float);

    std::cout << "Image: " << filename << " is size: " << imageWidth << "x" << imageHeight << "with " << bitsPerPixel << "bits per pixel" << "." << std::endl;
    std::cout << "Image Type: " << imageType << std::endl;
    std::cout << "Image component(which is used to step to the next pixel):" << bytespp << std::endl;

    //--Debug info end

    //////////////////////////////////////////////////////////////////////////
    //Create Buffer
    float* buffer_data = static_cast<float*>(buffer->map());

    //todo:review BITMAP Scanline is upside down,see also FreeImage Pixel Access Function for further details.

    for (auto y = 0; y < imageHeight; ++y)
    {
        //Note the scanline fetched by FreeImage is upside down--the first scanline corresponds to the buttom of the image!
        FLOAT *bits = reinterpret_cast<FLOAT *>(FreeImage_GetScanLine(bitmap, /*imageHeight - y - 1*/y));

        for (auto x = 0; x < imageWidth; ++x)
        {
            unsigned int buf_index = (imageWidth * y + x) * 4;

            //32bit image texture is linear.
            //Todo:validate the input color value, in case of being nan/inf
            TW_ASSERT(!isinf(buffer_data[buf_index]) && !isnan(buffer_data[buf_index]));
            TW_ASSERT(!isinf(buffer_data[buf_index + 1]) && !isnan(buffer_data[buf_index + 1]));
            TW_ASSERT(!isinf(buffer_data[buf_index + 2]) && !isnan(buffer_data[buf_index + 2]));


            static_cast<float>(bits[0]) = buffer_data[buf_index];
            static_cast<float>(bits[1]) = buffer_data[buf_index + 1];
            static_cast<float>(bits[2]) = buffer_data[buf_index + 2];
            static_cast<float>(bits[3]) = 1.f;
            // jump to next pixel
            bits += bytespp;
        }
    }

    buffer->unmap();
    FreeImage_Save(FIF_EXR, bitmap, filename.c_str());
    // Unload the 32-bit colour bitmap
    FreeImage_Unload(bitmap);
}

void ImageLoader::SaveLuminanceBufferToImage(optix::Buffer buffer, const std::string filename)
{
	//////////////////////////////////////////////////////////////////////////
	//Load Image using FreeImage

	// If we're here we have a known image format, so load the image into a bitap
	RTsize bufferSizeW, bufferSizeH;
	buffer->getSize(bufferSizeW, bufferSizeH);
	FIBITMAP* bitmap = FreeImage_AllocateT(FIT_RGBAF, bufferSizeW, bufferSizeH);

	//--Debug info
	// How many bits-per-pixel is the source image?
	int bitsPerPixel = FreeImage_GetBPP(bitmap);

	int imageWidth = FreeImage_GetWidth(bitmap);
	int imageHeight = FreeImage_GetHeight(bitmap);

	FREE_IMAGE_TYPE imageType = FreeImage_GetImageType(bitmap);
	int bytespp = FreeImage_GetLine(bitmap) / imageWidth / sizeof(float);

	std::cout << "Image: " << filename << " is size: " << imageWidth << "x" << imageHeight << "with " << bitsPerPixel << "bits per pixel" << "." << std::endl;
	std::cout << "Image Type: " << imageType << std::endl;
	std::cout << "Image component(which is used to step to the next pixel):" << bytespp << std::endl;

	//--Debug info end

	//////////////////////////////////////////////////////////////////////////
	//Create Buffer
	float* buffer_data = static_cast<float*>(buffer->map());

	//todo:review BITMAP Scanline is upside down,see also FreeImage Pixel Access Function for further details.

	for (auto y = 0; y < imageHeight; ++y)
	{
		//Note the scanline fetched by FreeImage is upside down--the first scanline corresponds to the buttom of the image!
		FLOAT *bits = reinterpret_cast<FLOAT *>(FreeImage_GetScanLine(bitmap, imageHeight - y - 1));

		for (auto x = 0; x < imageWidth; ++x)
		{
			unsigned int buf_index = (imageWidth * y + x) * 1;

			//32bit image texture is linear.
			static_cast<float>(bits[0]) = buffer_data[buf_index];
			static_cast<float>(bits[1]) = buffer_data[buf_index];
			static_cast<float>(bits[2]) = buffer_data[buf_index];
			static_cast<float>(bits[3]) = 1.f;
			// jump to next pixel
			bits += bytespp;
		}
	}

	buffer->unmap();
	FreeImage_Save(FIF_EXR, bitmap, filename.c_str());
	// Unload the 32-bit colour bitmap
	FreeImage_Unload(bitmap);
}

optix::TextureSampler ImageLoader::LoadImageTextureLinear(optix::Context & context, const std::string filename, const optix::float4 defaultColor)
{
	//////////////////////////////////////////////////////////////////////////
	//Setup texture sampler
	optix::TextureSampler sampler = context->createTextureSampler();
	sampler->setWrapMode(0, RT_WRAP_REPEAT);
	sampler->setWrapMode(1, RT_WRAP_REPEAT);
	sampler->setWrapMode(2, RT_WRAP_REPEAT);
	sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
	sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
	sampler->setMaxAnisotropy(1.0f);

	//////////////////////////////////////////////////////////////////////////
	//Load Image using FreeImage

	//Get extension name from filename string
	auto getFreeImageFormat = [&filename]()->FREE_IMAGE_FORMAT
	{
		// Get the filename extension                                                
		std::string::size_type extension_index = filename.find_last_of(".");
		std::string ext = extension_index != std::string::npos ?
			filename.substr(extension_index + 1) :
			std::string();
		std::locale loc;
		for (std::string::size_type i = 0; i < ext.length(); ++i)
			ext[i] = std::tolower(ext[i], loc);
		if (ext == "hdr")
			return FIF_HDR;
		if (ext == "exr")
			return FIF_EXR;
		if (ext == "png")
			return FIF_PNG;
		return FIF_UNKNOWN;
	};

	auto imageFormat = getFreeImageFormat();
	if (imageFormat == FIF_UNKNOWN)
	{
		std::cerr << "The format is neither HDR nor EXR, not supported..." << std::endl;
	}

	// If we're here we have a known image format, so load the image into a bitap
	FIBITMAP* bitmap = FreeImage_Load(imageFormat, filename.c_str());

	TW_ASSERT(bitmap != nullptr);

	//--Debug info
	// How many bits-per-pixel is the source image?
	int bitsPerPixel = FreeImage_GetBPP(bitmap);

	int imageWidth = FreeImage_GetWidth(bitmap);
	int imageHeight = FreeImage_GetHeight(bitmap);

	FREE_IMAGE_TYPE imageType = FreeImage_GetImageType(bitmap);
	int bytespp = FreeImage_GetLine(bitmap) / imageWidth / sizeof(float);

	std::cout << "Image: " << filename << " is size: " << imageWidth << "x" << imageHeight << "with " << bitsPerPixel << "bits per pixel" << "." << std::endl;
	std::cout << "Image Type: " << imageType << std::endl;
	std::cout << "Image component(which is used to step to the next pixel):" << bytespp << std::endl;

	//--Debug info end

	//////////////////////////////////////////////////////////////////////////
	//Create Buffer
	optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, imageWidth, imageHeight);
	float* buffer_data = static_cast<float*>(buffer->map());

	//todo:review BITMAP Scanline is upside down,see also FreeImage Pixel Access Function for further details.

	switch (imageType)
	{
	case FIT_RGBAF:
		for (auto y = 0; y < imageHeight; ++y)
		{
			//Note the scanline fetched by FreeImage is upside down--the first scanline corresponds to the buttom of the image!
			FLOAT *bits = reinterpret_cast<FLOAT *>(FreeImage_GetScanLine(bitmap, imageHeight - y - 1));

			for (auto x = 0; x < imageWidth; ++x)
			{
				unsigned int buf_index = (imageWidth * y + x) * 4;

				//32bit image texture is linear.
				//Todo:validate the input color value, in case of being nan/inf
				TW_ASSERT(!isinf(static_cast<float>(bits[0])) && !isnan(static_cast<float>(bits[0])));
				TW_ASSERT(!isinf(static_cast<float>(bits[1])) && !isnan(static_cast<float>(bits[1])));
				TW_ASSERT(!isinf(static_cast<float>(bits[2])) && !isnan(static_cast<float>(bits[2])));

				//note that for RGBAF/RGBF format, the pixel order is:RGB(A)
				buffer_data[buf_index] = static_cast<float>(bits[0]);
				buffer_data[buf_index + 1] = static_cast<float>(bits[1]);
				buffer_data[buf_index + 2] = static_cast<float>(bits[2]);
				buffer_data[buf_index + 3] = 1.0f;
				// jump to next pixel
				bits += bytespp;
			}
		}
		break;
	case FIT_RGBF:
		for (auto y = 0; y < imageHeight; ++y)
		{
			FLOAT *bits = reinterpret_cast<FLOAT *>(FreeImage_GetScanLine(bitmap, imageHeight - y - 1));
			//todo:review the code
			//when dealing with 32-bits image, note the actual struct read by FreeImage is FIRGBAF rather than FLOAT, thus the accessing code below should be written in consistent with the layout of FIRGBAF.
			for (auto x = 0; x < imageWidth; ++x)
			{
				unsigned int buf_index = (imageWidth * y + x) * 4;

				//32bit image texture is linear.
				//Todo:validate the input color value, in case of being nan/inf
				TW_ASSERT(!isinf(static_cast<float>(bits[0])) && !isnan(static_cast<float>(bits[0])));
				TW_ASSERT(!isinf(static_cast<float>(bits[1])) && !isnan(static_cast<float>(bits[1])));
				TW_ASSERT(!isinf(static_cast<float>(bits[2])) && !isnan(static_cast<float>(bits[2])));
				/*if (static_cast<float>(bits[0]) >= 20.f)
					std::cerr << "[Info]the loading image file contains pixel brighter than 20.f:" << static_cast<float>(bits[0]) << std::endl;
				if (static_cast<float>(bits[1]) >= 20.f)
					std::cerr << "[Info]the loading image file contains pixel brighter than 20.f:" << static_cast<float>(bits[1]) << std::endl;
				if (static_cast<float>(bits[2]) >= 20.f)
					std::cerr << "[Info]the loading image file contains pixel brighter than 20.f:" << static_cast<float>(bits[2]) << std::endl;*/

				buffer_data[buf_index] = static_cast<float>(bits[0]);
				buffer_data[buf_index + 1] = static_cast<float>(bits[1]);
				buffer_data[buf_index + 2] = static_cast<float>(bits[2]);
				buffer_data[buf_index + 3] = 1.0f;
				// jump to next pixel
				bits += bytespp;
			}
		}
		break;
	default:
		std::cerr << "Type of the image is not RGBAF, not supported yet..." << std::endl;
		break;
	}

	buffer->unmap();

	sampler->setBuffer(0u, 0u, buffer);
	sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);

	// Unload the 32-bit colour bitmap
	FreeImage_Unload(bitmap);

	return sampler;
}

optix::TextureSampler ImageLoader::LoadImageTextureGamma(optix::Context & context, const std::string filename, const optix::float4 defaultColor)
{
	//////////////////////////////////////////////////////////////////////////
	//Setup texture sampler
	optix::TextureSampler sampler = context->createTextureSampler();
	sampler->setWrapMode(0, RT_WRAP_REPEAT);
	sampler->setWrapMode(1, RT_WRAP_REPEAT);
	sampler->setWrapMode(2, RT_WRAP_REPEAT);
	sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
	sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
	sampler->setMaxAnisotropy(1.0f);

	//////////////////////////////////////////////////////////////////////////
	//Load Image using FreeImage

	//Get extension name from filename string
	auto getFreeImageFormat = [&filename]()->FREE_IMAGE_FORMAT
	{
		// Get the filename extension                                                
		std::string::size_type extension_index = filename.find_last_of(".");
		std::string ext = extension_index != std::string::npos ?
			filename.substr(extension_index + 1) :
			std::string();
		std::locale loc;
		for (std::string::size_type i = 0; i < ext.length(); ++i)
			ext[i] = std::tolower(ext[i], loc);

		if (ext == "png")
			return FIF_PNG;
		if (ext == "tga")
			return FIF_TARGA;
		return FIF_UNKNOWN;
	};

	auto imageFormat = getFreeImageFormat();
	if (imageFormat == FIF_UNKNOWN)
	{
		std::cerr << "The format is not PNG/TGA but need to be treated as Gamma, not supported..." << std::endl;
	}

	// If we're here we have a known image format, so load the image into a bitap
	FIBITMAP* bitmap = FreeImage_Load(imageFormat, filename.c_str());

	TW_ASSERT(bitmap != nullptr);

	//--Debug info
	// How many bits-per-pixel is the source image?
	int bitsPerPixel = FreeImage_GetBPP(bitmap);

	int imageWidth = FreeImage_GetWidth(bitmap);
	int imageHeight = FreeImage_GetHeight(bitmap);

	FREE_IMAGE_TYPE imageType = FreeImage_GetImageType(bitmap);
	int bytespp = FreeImage_GetLine(bitmap) / imageWidth / sizeof(BYTE);

	std::cout << "Image: " << filename << " is size: " << imageWidth << "x" << imageHeight << "with " << bitsPerPixel << "bits per pixel" << "." << std::endl;
	std::cout << "Image Type: " << imageType << std::endl;
	std::cout << "Image component(which is used to step to the next pixel):" << bytespp << std::endl;

	//--Debug info end

	//////////////////////////////////////////////////////////////////////////
	//Create Buffer
	optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, imageWidth, imageHeight);
	float* buffer_data = static_cast<float*>(buffer->map());

	//todo:review BITMAP Scanline is upside down,see also FreeImage Pixel Access Function for further details.

	//see also:https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_sRGB_decode.txt
	auto convert_sRGBToLinear = [](const float &src)->float
	{
		if (src <= 0.f)
			return 0;
		if (src >= 1.f)
			return 1.f;
		if (src <= 0.04045f)
			return src / 12.92f;
		return pow((src + 0.055f) / 1.055f, 2.4f);
	};


	switch (imageType)
	{
	case FIT_BITMAP:
		for (auto y = 0; y < imageHeight; ++y)
		{
			//Note the scanline fetched by FreeImage is upside down--the first scanline corresponds to the buttom of the image!
			BYTE *bits = reinterpret_cast<BYTE *>(FreeImage_GetScanLine(bitmap, imageHeight - y - 1));

			for (auto x = 0; x < imageWidth; ++x)
			{
				//RGB components in Texture Buffer(A doesn't work,just for alignment)
				unsigned int buf_index = (imageWidth * y + x) * 4;

				//Todo:validate the input color value, in case of being nan/inf

				//note that for non-RGB(A)F format,the pixel order is OS dependent, thus the enum provided by FreeImage should be used insted.

				//Gamma Correction needs to be performed:
				buffer_data[buf_index]     = convert_sRGBToLinear(bits[FI_RGBA_RED] / 255.f);
				buffer_data[buf_index + 1] = convert_sRGBToLinear(bits[FI_RGBA_GREEN] / 255.f);
				buffer_data[buf_index + 2] = convert_sRGBToLinear(bits[FI_RGBA_BLUE] / 255.f);
				buffer_data[buf_index + 3] = 1.0f;//discard the alpha channel(even exists)
				// jump to next pixel
				bits += bytespp;
			}
		}
		break;
	default:
		std::cerr << "Type of the image is not BITMAP, not supported yet..." << std::endl;
		break;
	}

	buffer->unmap();

	sampler->setBuffer(0u, 0u, buffer);
	sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);

	// Unload the 32-bit colour bitmap
	FreeImage_Unload(bitmap);

	return sampler;
}

optix::TextureSampler ImageLoader::LoadImageTextureAlpha(optix::Context & context, const std::string filename, const optix::float4 defaultColor)
{
	//////////////////////////////////////////////////////////////////////////
	//Setup texture sampler
	optix::TextureSampler sampler = context->createTextureSampler();
	sampler->setWrapMode(0, RT_WRAP_REPEAT);
	sampler->setWrapMode(1, RT_WRAP_REPEAT);
	sampler->setWrapMode(2, RT_WRAP_REPEAT);
	sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
	sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
	sampler->setMaxAnisotropy(1.0f);

	//////////////////////////////////////////////////////////////////////////
	//Load Image using FreeImage

	//Get extension name from filename string
	auto getFreeImageFormat = [&filename]()->FREE_IMAGE_FORMAT
	{
		// Get the filename extension                                                
		std::string::size_type extension_index = filename.find_last_of(".");
		std::string ext = extension_index != std::string::npos ?
			filename.substr(extension_index + 1) :
			std::string();
		std::locale loc;
		for (std::string::size_type i = 0; i < ext.length(); ++i)
			ext[i] = std::tolower(ext[i], loc);

		if (ext == "png")
			return FIF_PNG;
		if (ext == "tga")
			return FIF_TARGA;
		return FIF_UNKNOWN;
	};

	auto imageFormat = getFreeImageFormat();
	if (imageFormat == FIF_UNKNOWN)
	{
		std::cerr << "The format is not PNG/TGA, not supported..." << std::endl;
	}

	// If we're here we have a known image format, so load the image into a bitap
	FIBITMAP* bitmap = FreeImage_Load(imageFormat, filename.c_str());

	TW_ASSERT(bitmap != nullptr);

	//--Debug info
	// How many bits-per-pixel is the source image?
	int bitsPerPixel = FreeImage_GetBPP(bitmap);

	int imageWidth = FreeImage_GetWidth(bitmap);
	int imageHeight = FreeImage_GetHeight(bitmap);

	FREE_IMAGE_TYPE imageType = FreeImage_GetImageType(bitmap);
	int bytespp = FreeImage_GetLine(bitmap) / imageWidth / sizeof(BYTE);

	std::cout << "Image: " << filename << " is size: " << imageWidth << "x" << imageHeight << "with " << bitsPerPixel << "bits per pixel" << "." << std::endl;
	std::cout << "Image Type: " << imageType << std::endl;
	std::cout << "Image component(which is used to step to the next pixel):" << bytespp << std::endl;

	//--Debug info end

	//////////////////////////////////////////////////////////////////////////
	//Create Buffer
	optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, imageWidth, imageHeight);
	float* buffer_data = static_cast<float*>(buffer->map());

	//todo:review BITMAP Scanline is upside down,see also FreeImage Pixel Access Function for further details.



	switch (imageType)
	{
	case FIT_BITMAP:
		for (auto y = 0; y < imageHeight; ++y)
		{
			//Note the scanline fetched by FreeImage is upside down--the first scanline corresponds to the buttom of the image!
			BYTE *bits = reinterpret_cast<BYTE *>(FreeImage_GetScanLine(bitmap, imageHeight - y - 1));

			for (auto x = 0; x < imageWidth; ++x)
			{
				//only A component
				unsigned int buf_index = (imageWidth * y + x) * 4;

				//Todo:validate the input color value, in case of being nan/inf
				/*TW_ASSERT(!isinf(static_cast<float>(bits[0])) && !isnan(static_cast<float>(bits[0])));
				TW_ASSERT(!isinf(static_cast<float>(bits[1])) && !isnan(static_cast<float>(bits[1])));
				TW_ASSERT(!isinf(static_cast<float>(bits[2])) && !isnan(static_cast<float>(bits[2])));*/

				//if (bits[0] == 0)
				//	std::cout << static_cast<float>(bits[0]) << std::endl;

				buffer_data[buf_index]     = (bits[FI_RGBA_RED] / 255.f);
				buffer_data[buf_index + 1] = (bits[FI_RGBA_GREEN] / 255.f);
				buffer_data[buf_index + 2] = (bits[FI_RGBA_BLUE] / 255.f);
				buffer_data[buf_index + 3] = 1.0f;//discard the alpha channel(even exists)
				// jump to next pixel
				bits += bytespp;
			}
		}
		break;
	default:
		std::cerr << "Type of the image is not BITMAP, not supported yet..." << std::endl;
		break;
	}

	buffer->unmap();

	sampler->setBuffer(0u, 0u, buffer);
	sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_LINEAR);

	// Unload the 32-bit colour bitmap
	FreeImage_Unload(bitmap);

	return sampler;
}
