function IMAGES = whitener(IMAGES)

    N = size(IMAGES, 1);
    M = size(IMAGES, 3);

    [fx fy]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);
    rho=sqrt(fx.*fx+fy.*fy);
    f_0=0.4*N;
    filt=rho.*exp(-(rho/f_0).^4);

    for i=1:M
        image=IMAGES(:,:,i);
        If=fft2(image);
        imagew=real(ifft2(If.*fftshift(filt)));
        IMAGES_tmp(:,i)=reshape(imagew,N^2,1);
    end

    IMAGES_tmp=sqrt(0.1)*IMAGES_tmp/sqrt(mean(var(IMAGES_tmp)));
    IMAGES = reshape(IMAGES_tmp, size(IMAGES));
end